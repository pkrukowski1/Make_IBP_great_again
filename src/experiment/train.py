import logging
import numpy as np
import time
import torch
import wandb
import json
import os

from omegaconf import DictConfig
from hydra.utils import instantiate
from tqdm import tqdm

from utils.fabric import setup_fabric
from utils.hydra import extract_output_dir
from experiment.utils import get_dataloader, verify_point, check_correct_prediction, get_eps
from train import Trainer

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def evaluate_split(split_name, dataloader, fabric, trainer, method, config):
    log.info(f"Evaluating on {split_name} set")
    total_loss = 0.0
    total_samples = 0

    all_bounds = []
    processing_times = []
    verified_points = []
    batch_results = []

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(dataloader):
            X = fabric.to_device(X)
            y = fabric.to_device(y)

            loss, bounds = trainer.forward(X, y)
            total_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

            bound_width = (bounds.upper - bounds.lower)
            all_bounds.append(bound_width.detach().cpu())

            # --- Robustness verification ---
            _, y_pred = check_correct_prediction(method.module, X, y)
            base_eps = torch.ones_like(X)
            base_eps = fabric.to_device(base_eps)
            eps = get_eps(config, base_eps)

            start_time = time.time()
            int_output_bounds = method.forward(X, y, eps)
            batch_time = time.time() - start_time
            avg_time_per_image = batch_time / X.size(0)
            processing_times.append(avg_time_per_image)

            lb = int_output_bounds.lower
            ub = int_output_bounds.upper
            output_bounds_length = torch.max(ub - lb, dim=-1).values.item()
            verified = verify_point(int_output_bounds, y_pred, y)
            verified_points.extend([] if verified is None else [verified])

            batch_results.append({
                "batch_idx": batch_idx,
                "output_bounds_length": output_bounds_length,
                "verified_point": verified,
                "avg_time_per_image": avg_time_per_image
            })

    avg_loss = total_loss / total_samples
    all_bounds_tensor = torch.cat(all_bounds)
    flat_bounds = all_bounds_tensor.flatten().numpy()
    avg_bound = all_bounds_tensor.mean().item()
    max_bound = all_bounds_tensor.max().item()
    min_bound = all_bounds_tensor.min().item()
    overall_avg_time = np.mean(processing_times)
    overall_verified_points = np.sum(verified_points) / len(dataloader)

    wandb.log({
        f"{split_name}/avg_loss": avg_loss,
        f"{split_name}/avg_bound_width": avg_bound,
        f"{split_name}/max_bound_width": max_bound,
        f"{split_name}/min_bound_width": min_bound,
        f"{split_name}/bound_width_hist": wandb.Histogram(flat_bounds),
        f"{split_name}/overall_avg_processing_time_per_image": overall_avg_time,
        f"{split_name}/overall_verified_points": overall_verified_points
    })

    return {
        "avg_loss": avg_loss,
        "avg_bound_width": avg_bound,
        "max_bound_width": max_bound,
        "min_bound_width": min_bound,
        "overall_avg_time": overall_avg_time,
        "overall_verified_points": overall_verified_points,
        "batch_results": batch_results
    }

def run(config: DictConfig):
    wandb.init(project=os.environ['WANDB_PROJECT'], config=dict(config))

    log.info(f'Launching Fabric')
    fabric = setup_fabric(config)

    log.info(f'Building model')
    model = fabric.setup(instantiate(config.network))

    log.info(f'Setting up training method')
    method = instantiate(config.method)(model)

    log.info(f'Setting up dataloaders')
    train_loader = get_dataloader(config.dataset, fabric, split='train')
    val_loader   = get_dataloader(config.dataset, fabric, split='val')
    test_loader  = get_dataloader(config.dataset, fabric, split='test')

    extracted_output_dir = extract_output_dir(config)
    output_file = f"{extracted_output_dir}/training_log.json"

    log.info(f'Initializing the trainer')
    trainer = Trainer(
        method=method,
        start_epoch=config.training.start_warmup_epoch,
        end_epoch=config.training.end_warmup_epoch,
    )

    optimizer = torch.optim.Adam(trainer.method.module.parameters(), lr=config.training.lr)

    epochs = config.training.epochs
    use_scheduler = config.training.get("use_scheduler", False)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    epoch_logs = []

    for epoch in range(epochs):
        trainer.set_epoch(epoch)
        epoch_loss = 0.0
        total_samples = 0

        log.info(f"Epoch {epoch+1}/{epochs} — epsilon: {trainer.current_epsilon:.5f}, kappa: {trainer.current_kappa:.5f}")

        for batch_idx, (X, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} — Training")):
            X = fabric.to_device(X)
            y = fabric.to_device(y)

            loss, bounds = trainer.forward(X, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

            bound_width = (bounds.upper - bounds.lower)
            avg_bound_width = bound_width.mean().item()
            max_bound_width = bound_width.max().item()
            min_bound_width = bound_width.min().item()
            flat_bound_width = bound_width.detach().cpu().flatten().numpy()

            wandb.log({
                "train/batch_loss": loss.item(),
                "train/epsilon": trainer.current_epsilon,
                "train/kappa": trainer.current_kappa,
                "train/epoch": epoch,
                "train/batch_bound_avg": avg_bound_width,
                "train/batch_bound_max": max_bound_width,
                "train/batch_bound_min": min_bound_width,
                "train/bound_width_hist": wandb.Histogram(flat_bound_width)
            })

        avg_train_loss = epoch_loss / total_samples
        log.info(f"Epoch {epoch+1} — Train Loss: {avg_train_loss:.6f}")

        if use_scheduler:
            scheduler.step(avg_train_loss)

        val_stats = evaluate_split("val", val_loader, fabric, trainer, method, config)

        wandb.log({
            "train/epoch_loss": avg_train_loss,
            "train/epoch": epoch,
        })

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_epsilon": trainer.current_epsilon,
            "train_kappa": trainer.current_kappa,
            "val_loss": val_stats["avg_loss"],
            "val_avg_bound_width": val_stats["avg_bound_width"],
            "val_max_bound_width": val_stats["max_bound_width"],
            "val_min_bound_width": val_stats["min_bound_width"],
            "val_avg_time_per_image": val_stats["overall_avg_time"],
            "val_verified_points": val_stats["overall_verified_points"]
        })

    test_stats = evaluate_split("test", test_loader, fabric, trainer, method, config)

    output = {
        "training_log": epoch_logs,
        "final_test_stats": test_stats
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)

    wandb.finish()
    log.info(f"Training complete. Logs saved to {output_file}")

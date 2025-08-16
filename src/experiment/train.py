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
from torch.utils.data import DataLoader
from lightning.fabric import Fabric
from typing import Any, Dict, List

from utils.fabric import setup_fabric
from utils.hydra import extract_output_dir
from experiment.utils import get_eps, compute_verified_error, pgd_linf_attack
from dataset.factory import DatasetFactory
from train import Trainer

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def evaluate_split(
    split_name: str,
    dataloader: DataLoader,
    fabric: Fabric,
    trainer: Trainer,
    method: Any,
    config: DictConfig
) -> Dict[str, float]:
    """
    Evaluate model performance on a given dataset split, logging both
    standard metrics and robust verification metrics to W&B.
    """
    log.info(f"Evaluating on {split_name} set")

    method.module.eval()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_pgd_wrong = 0
    total_verified_error: List[float] = []

    all_bounds = []
    processing_times = []

    with torch.no_grad():
        for _, (X, y) in enumerate(tqdm(dataloader)):
            X = fabric.to_device(X)
            y = fabric.to_device(y)

            # Forward pass to get natural loss and bounds
            loss, bounds = trainer.forward(X, y)
            total_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

            # Clean predictions
            y_pred_clean = torch.argmax(method.module(X), dim=1)
            correct = (y_pred_clean == y).sum().item()
            total_correct += correct

            # Robustness verification
            base_eps = torch.ones_like(X)
            eps = get_eps(config, fabric.to_device(base_eps))
            start_time = time.time()
            int_output_bounds = method.forward(X, y, eps)
            batch_time = time.time() - start_time
            processing_times.append(batch_time / X.size(0))

            verified_err_batch = compute_verified_error(int_output_bounds, y)
            total_verified_error.append(verified_err_batch)

            # PGD-200 attack
            adv_x = pgd_linf_attack(method.module, X, y, eps=eps.item(), steps=200)
            y_pred_pgd = torch.argmax(method.module(adv_x), dim=1)
            total_pgd_wrong += (y_pred_pgd != y).sum().item()

            # Track bound widths
            bound_width = (bounds.upper - bounds.lower)
            all_bounds.append(bound_width.detach().cpu())

    # Aggregate results
    avg_loss = total_loss / total_samples
    clean_error = 100.0 * (1.0 - (total_correct / total_samples))
    pgd_error = 100.0 * (total_pgd_wrong / total_samples)
    verified_error = np.mean(total_verified_error)

    all_bounds_tensor = torch.cat(all_bounds)
    flat_bounds = all_bounds_tensor.flatten().numpy()
    avg_bound = all_bounds_tensor.mean().item()
    max_bound = all_bounds_tensor.max().item()
    min_bound = all_bounds_tensor.min().item()
    overall_avg_time = np.mean(processing_times)

    # Log to W&B
    wandb.log({
        f"{split_name}/avg_loss": avg_loss,
        f"{split_name}/clean_error": clean_error,
        f"{split_name}/pgd_error": pgd_error,
        f"{split_name}/verified_error": verified_error,
        f"{split_name}/avg_bound_width": avg_bound,
        f"{split_name}/max_bound_width": max_bound,
        f"{split_name}/min_bound_width": min_bound,
        f"{split_name}/bound_width_hist": wandb.Histogram(flat_bounds),
        f"{split_name}/overall_avg_processing_time_per_image": overall_avg_time,
    })

    # Console logging
    print(f"[{split_name.upper()}] Avg Loss: {avg_loss:.6f}, "
          f"Clean Err: {clean_error:.2f}%, "
          f"PGD-200 Err: {pgd_error:.2f}%, "
          f"Verified Err: {verified_error:.2f}%, "
          f"Avg Bound Width: {avg_bound:.6f}, "
          f"Max Bound Width: {max_bound:.6f}, Min Bound Width: {min_bound:.6f}, "
          f"Avg Proc Time/Image: {overall_avg_time:.6f}s")

    return {
        "avg_loss": avg_loss,
        "clean_error": clean_error,
        "pgd_error": pgd_error,
        "verified_error": verified_error,
        "avg_bound_width": avg_bound,
        "max_bound_width": max_bound,
        "min_bound_width": min_bound,
        "overall_avg_time": overall_avg_time
    }


def run(config: DictConfig) -> None:
    """
    Main training and evaluation loop with metric logging.
    """
    wandb.init(project=os.environ['WANDB_PROJECT'], config=dict(config))

    log.info(f'Launching Fabric')
    fabric = setup_fabric(config)

    log.info(f'Building model')
    model = fabric.setup(instantiate(config.network))

    log.info(f'Setting up training method')
    method = instantiate(config.method)(model)

    log.info(f'Setting up dataloaders')

    train_dataset, val_dataset, test_dataset = DatasetFactory.get_dataset(
        dataset_type=config.dataset.type,
        data_path=config.dataset.data_path,
        flatten=config.dataset.flatten,
        split_ratio=config.dataset.split_ratio,
        seed=config.dataset.seed,
        download=config.dataset.download
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.train_batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataset.val_batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.dataset.test_batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
    )

    train_loader, val_loader, test_loader = fabric.setup_dataloaders(
        train_loader, val_loader, test_loader
    )

    extracted_output_dir = extract_output_dir(config)
    output_file = f"{extracted_output_dir}/training_log.json"

    log.info(f'Initializing the trainer')
    trainer = Trainer(
        method=method,
        warmup_start_epoch=config.training.warmup_start_epoch,
        warmup_end_epoch=config.training.warmup_end_epoch,
        schedule_epochs=config.training.schedule_epochs_after_warmup,
        num_batches_per_epoch=len(train_loader)
    )

    optimizer = torch.optim.Adam(trainer.method.module.parameters(), lr=config.training.lr)

    epochs: int = config.training.epochs
    use_scheduler: bool = config.training.get("use_scheduler", False)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    epoch_logs: List[Dict[str, Any]] = []
    best_val_loss: float = float("inf")
    checkpoint_path: str = os.path.join(extracted_output_dir, "best_model.pth")

    for epoch in range(epochs):
        trainer.set_epoch(epoch)
        epoch_loss = 0.0
        total_samples = 0
        total_correct = 0
        total_pgd_wrong = 0
        total_verified_error: List[float] = []

        log.info(f"Epoch {epoch+1}/{epochs} — epsilon: {trainer.current_epsilon:.5f}, kappa: {trainer.current_kappa:.5f}")
        print(f"\nEpoch {epoch+1}/{epochs} — epsilon: {trainer.current_epsilon:.5f}, kappa: {trainer.current_kappa:.5f}")

        trainer.method.module.train()
        for batch_idx, (X, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} — Training")):
            X = fabric.to_device(X)
            y = fabric.to_device(y)

            loss, bounds = trainer.forward(X, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Clean accuracy
            y_pred = torch.argmax(trainer.method.module(X), dim=1)
            correct = (y_pred == y).sum().item()
            total_correct += correct
            batch_accuracy = correct / X.size(0)

            # Verified error (batch)
            eps = get_eps(config, fabric.to_device(torch.ones_like(X)))
            int_output_bounds = trainer.method.forward(X, y, eps)
            verified_err_batch = compute_verified_error(int_output_bounds, y)
            total_verified_error.append(verified_err_batch)

            # PGD error (batch)
            adv_x = pgd_linf_attack(trainer.method.module, X, y, eps=eps.item(), steps=200)
            y_pred_pgd = torch.argmax(trainer.method.module(adv_x), dim=1)
            total_pgd_wrong += (y_pred_pgd != y).sum().item()

            robust_accuracy = 1.0 - (verified_err_batch / 100.0)

            # Logging to console
            print(f"[Batch {batch_idx+1}] Loss: {loss.item():.6f} | Acc: {batch_accuracy:.4f} | "
                  f"PGD Err: {100*(1-robust_accuracy):.2f}% | Verified Err: {verified_err_batch:.2f}%")

            epoch_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

            # Bound widths
            bound_width = (bounds.upper - bounds.lower)
            avg_bound_width = bound_width.mean().item()
            max_bound_width = bound_width.max().item()
            min_bound_width = bound_width.min().item()
            flat_bound_width = bound_width.detach().cpu().flatten().numpy()

            # Batch logs
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/epsilon": trainer.current_epsilon,
                "train/kappa": trainer.current_kappa,
                "train/epoch": epoch,
                "train/batch_clean_error": 100.0 * (1.0 - batch_accuracy),
                "train/batch_pgd_error": 100.0 * (y_pred_pgd != y).float().mean().item(),
                "train/batch_verified_error": verified_err_batch,
                "train/batch_bound_avg": avg_bound_width,
                "train/batch_bound_max": max_bound_width,
                "train/batch_bound_min": min_bound_width,
                "train/bound_width_hist": wandb.Histogram(flat_bound_width)
            })

        # Epoch aggregates
        avg_train_loss = epoch_loss / total_samples
        train_clean_error = 100.0 * (1.0 - (total_correct / total_samples))
        train_pgd_error = 100.0 * (total_pgd_wrong / total_samples)
        train_verified_error = np.mean(total_verified_error)

        if use_scheduler:
            scheduler.step(avg_train_loss)

        # Validation
        val_stats = evaluate_split("val", val_loader, fabric, trainer, method, config)

        # Save best model
        if val_stats["avg_loss"] < best_val_loss:
            best_val_loss = val_stats["avg_loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": trainer.method.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "epsilon": trainer.current_epsilon,
                "kappa": trainer.current_kappa
            }, checkpoint_path)
            log.info(f"Saved new best model at epoch {epoch+1} with val loss {best_val_loss:.6f}")
            print(f"Saved new best model at epoch {epoch+1} with val loss {best_val_loss:.6f}")

        # Epoch logs
        wandb.log({
            "train/epoch_loss": avg_train_loss,
            "train/epoch_clean_error": train_clean_error,
            "train/epoch_pgd_error": train_pgd_error,
            "train/epoch_verified_error": train_verified_error,
            "train/epoch": epoch,
        })

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_clean_error": train_clean_error,
            "train_pgd_error": train_pgd_error,
            "train_verified_error": train_verified_error,
            "train_epsilon": trainer.current_epsilon,
            "train_kappa": trainer.current_kappa,
            "val_loss": val_stats["avg_loss"],
            "val_clean_error": val_stats["clean_error"],
            "val_pgd_error": val_stats["pgd_error"],
            "val_verified_error": val_stats["verified_error"],
            "val_avg_bound_width": val_stats["avg_bound_width"],
            "val_max_bound_width": val_stats["max_bound_width"],
            "val_min_bound_width": val_stats["min_bound_width"],
            "val_avg_time_per_image": val_stats["overall_avg_time"],
        })

    # Test evaluation
    test_stats = evaluate_split("test", test_loader, fabric, trainer, method, config)

    output = {
        "training_log": epoch_logs,
        "final_test_stats": test_stats
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)

    wandb.finish()
    log.info(f"Training complete. Logs saved to {output_file}")
    print(f"Training complete. Logs saved to {output_file}")

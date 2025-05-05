import logging
import numpy as np
import time
import torch
import wandb
import json
import os

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm
from utils.fabric import setup_fabric
from utils.hydra import extract_output_dir
from experiment.utils import get_dataloader, verify_point

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def run(config: DictConfig):
    wandb.init(project=os.environ['WANDB_PROJECT'], config=OmegaConf.to_container(config, resolve=True))
    
    log.info('Launching Fabric')
    fabric = setup_fabric(config)

    log.info('Building model')
    model = fabric.setup(instantiate(config.network))

    log.info('Setting up method')
    method = instantiate(config.method)(model)

    log.info('Setting up dataloaders')
    dataloader = get_dataloader(config, fabric)

    extracted_output_dir = extract_output_dir(config)
    output_file = f"{extracted_output_dir}/output_bounds.json"
    
    max_eps = float(config.method.plugins[0].epsilon)
    eps_list = np.linspace(1e-8, max_eps, 20)

    all_results = []

    for eps in eps_list:
        log.info(f"Running for epsilon = {eps:.2e}")
        config.method.plugins[0].epsilon = float(eps)

        processing_times = []
        verified_points = []
        batch_results = []

        for batch_idx, (X, y) in enumerate(tqdm(dataloader, desc=f"Epsilon {eps:.2e}")):
            X = fabric.to_device(X)
            y = fabric.to_device(y)
            
            start_time = time.time()
            int_output_bounds = method.forward(X, y)
            batch_time = time.time() - start_time

            lb = int_output_bounds.lower
            ub = int_output_bounds.upper

            batch_size = X.size(0)
            avg_time_per_image = batch_time / batch_size
            processing_times.append(avg_time_per_image)

            output_bounds_length = torch.max(ub - lb, dim=-1).values.item()
            verified = verify_point(int_output_bounds, y)
            verified_points.extend([] if verified is None else [verified])

            wandb.log({
                "epsilon": eps,
                "batch_idx": batch_idx,
                "output_bounds_length": output_bounds_length,
                "avg_processing_time_per_image": avg_time_per_image,
                "correctly_verified_point": verified,
                "batch_size": batch_size
            })

            batch_results.append({
                "batch_idx": batch_idx,
                "output_bounds_length": output_bounds_length,
                "verified_point": verified,
                "avg_time_per_image": avg_time_per_image
            })

        # Compute overall metrics for this epsilon
        overall_avg_time = float(np.mean(processing_times)) if processing_times else 0.0
        overall_verified_points = float(np.sum(verified_points) / len(dataloader)) if verified_points else 0.0

        stats = {
            "epsilon": float(eps),
            "overall_avg_processing_time_per_image": overall_avg_time,
            "overall_verified_points_percent": 100 * overall_verified_points,
            "total_batches_processed": batch_idx + 1,
            "batches": batch_results
        }

        all_results.append(stats)

        wandb.log({
            "epsilon": eps,
            "overall_avg_processing_time_per_image": overall_avg_time,
            "overall_verified_points": overall_verified_points
        })

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    wandb.finish()
    log.info(f"Processing complete. Results saved to {output_file}")

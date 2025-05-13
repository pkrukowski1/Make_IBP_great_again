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
from experiment.utils import get_dataloader, verify_point, add_salt_and_pepper, save_deteriotated_image, \
    check_correct_prediction, get_eps

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def run(config: DictConfig):
    wandb.init(project=os.environ['WANDB_PROJECT'], config=dict(config))
    
    log.info(f'Launching Fabric')
    fabric = setup_fabric(config)

    log.info(f'Building model')
    model = fabric.setup(instantiate(config.network))

    log.info(f'Setting up method')
    method = instantiate(config.method)(model)

    log.info(f'Setting up dataloaders')
    dataloader = get_dataloader(config, fabric)

    extracted_output_dir = extract_output_dir(config)
    output_file = f"{extracted_output_dir}/output_bounds.json"
    
    processing_times = []
    verified_points = []
    batch_results = []
    
    for batch_idx, (X, y) in enumerate(tqdm(dataloader, desc="Processing batches")):
        X = fabric.to_device(X)
        y = fabric.to_device(y)
        base_eps = torch.ones_like(X)
        base_eps = fabric.to_device(base_eps)

        eps = get_eps(config, base_eps)

        X = add_salt_and_pepper(X, amount=config.exp.amount, salt_vs_pepper=config.exp.salt_vs_pepper)
        if config.exp.save_images:
            save_deteriotated_image(
                x=X,
                flatten=config.dataset.dataset.flatten,
                folder=extracted_output_dir,
                dataset=config.dataset.dataset._target_
            )
        
        _, y_pred = check_correct_prediction(method.module, X, y)
        
        start_time = time.time()
        
        int_output_bounds = method.forward(X, y, eps)
        log.info(f"Batch {batch_idx+1}, Output Bounds: {int_output_bounds}")
        
        # Calculate processing time
        batch_time = time.time() - start_time
        batch_size = X.size(0)
        avg_time_per_image = batch_time / batch_size
        processing_times.append(avg_time_per_image)
        
        # Convert output_bounds to numpy for logging/saving
        lb = int_output_bounds.lower
        ub = int_output_bounds.upper
        
        # Calculate metrics
        output_bounds_length = torch.max(ub - lb, dim=-1).values.item()
        verified = verify_point(int_output_bounds, y_pred, y)
        print(f"Batch {batch_idx+1}, Verified: {verified}")
        verified_points.extend([] if verified is None else [verified])
        
        # Log to wandb
        wandb.log({
            "batch_idx": batch_idx,
            "output_bounds_length": output_bounds_length,
            "avg_processing_time_per_image": avg_time_per_image,
            "correctly_verified_point": verified,
            "batch_size": batch_size
        })
        
        # Collect batch metrics
        batch_results.append({
            "batch_idx": batch_idx,
            "output_bounds_length": output_bounds_length,
            "verified_point": verified,
            "avg_time_per_image": avg_time_per_image
        })
            
    # Calculate overall metrics
    overall_avg_time = np.mean(processing_times)
    overall_verified_points = np.sum(verified_points) / len(dataloader)
    wandb.log({
        "overall_avg_processing_time_per_image": overall_avg_time,
        "overall_verified_points": overall_verified_points
    })
    
    # Prepare overall statistics
    overall_stats = {
        "overall_avg_processing_time_per_image": overall_avg_time,
        "overall_verified_points_percent": 100 * overall_verified_points,
        "total_batches_processed": batch_idx + 1
    }
    
    # Save to JSON file
    output_data = {
        "batches": batch_results,
        "overall_statistics": overall_stats
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    # Finish wandb run
    wandb.finish()
    
    log.info(f"Processing complete. Results saved to {output_file}")
    log.info(f"Overall average processing time per image: {overall_avg_time:.6f} seconds")
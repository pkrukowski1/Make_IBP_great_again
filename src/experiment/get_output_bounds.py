import logging
import numpy as np
import time
import torch
import wandb
import torch.nn.functional as F
import json

from omegaconf import DictConfig
from hydra.utils import instantiate
from tqdm import tqdm
from utils.fabric import setup_fabric
from utils.hydra import extract_output_dir
from method.interval_arithmetic import Interval

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def squeeze_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Squeeze batch dimension if batch size is 1, else return unchanged."""
    return tensor.squeeze(0) if tensor.size(0) == 1 else tensor

def get_dataloader(config: DictConfig, fabric) -> torch.utils.data.DataLoader:
    """
    Initializes and returns a dataloader using the provided configuration and fabric.
    Args:
        config (dict): A configuration object containing the dataset settings.
        fabric (object): An object responsible for setting up dataloaders.
    Returns:
        DataLoader: A dataloader instance prepared using the specified configuration and fabric.
    """
    return fabric.setup_dataloaders(instantiate(config.dataset))

def verify_point(output_bounds: Interval, label: torch.Tensor) -> float:
    """
    Verifies whether the predicted label for a given point is robustly classified
    based on the provided output bounds.
    Args:
        output_bounds (Interval): An object representing the interval bounds of the
            model's output logits. It should provide methods to access the midpoint,
            lower bounds, and upper bounds of the logits.
        label (torch.Tensor): A tensor containing the ground truth label for the
            input point.
    Returns:
        float: A verification result where 1.0 indicates that the point is verified.
    """
    with torch.no_grad():
        logits = output_bounds.midpoint()
        preds = F.softmax(logits, dim=-1)
        y_pred = torch.argmax(preds, dim=-1)

        lower_bound = output_bounds.lower
        upper_bound = output_bounds.upper

        lower_bound = squeeze_batch_dim(lower_bound)
        upper_bound = squeeze_batch_dim(upper_bound)
        verified = None

        if y_pred == label:
            verified = 0.0
            lower_bound_gt = lower_bound[y_pred]
            upper_bound_non_gt = upper_bound[torch.arange(upper_bound.size(0), device=upper_bound.device) != y_pred]
            if (lower_bound_gt > upper_bound_non_gt).all():
                verified = 1.0

        return verified

def run(config: DictConfig):
    # Initialize wandb
    wandb.init(project="image_processing", config=dict(config))
    
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
        
        start_time = time.time()
        
        int_output_bounds = method.forward(X, y)
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
        verified = verify_point(int_output_bounds, y)
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
    overall_verified_points = np.mean(verified_points)
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
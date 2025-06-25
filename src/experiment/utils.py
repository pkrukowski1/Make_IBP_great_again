import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os
import uuid
from typing import Tuple

from method.interval_arithmetic import Interval
from method.method_plugin_abc import MethodPluginABC

from omegaconf import DictConfig
from hydra.utils import instantiate


def squeeze_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Squeeze batch dimension if batch size is 1, else return unchanged."""
    return tensor.squeeze(0) if tensor.size(0) == 1 else tensor

def get_dataloader(config: DictConfig, fabric, split: str = None) -> DataLoader:
    """
    Initializes and returns a dataloader using the provided configuration and fabric.
    Args:
        config (dict): A configuration object containing the dataset settings.
        fabric (object): An object responsible for setting up dataloaders.
        split (str, optional): The dataset split to be used (e.g., 'train', 'val', 'test').
            If provided, it will be used to instantiate the dataset.
    Returns:
        DataLoader: A dataloader instance prepared using the specified configuration and fabric.
    """
    if split is not None:
        return fabric.setup_dataloaders(instantiate(config.dataset[split]))
    else:
        return fabric.setup_dataloaders(instantiate(config.dataset))

def verify_point(output_bounds: Interval, y_pred: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
    """
    Verifies robust classification only for correctly predicted samples.

    Args:
        output_bounds (Interval): Contains `.lower` and `.upper` bounds (B x C).
        y_pred (torch.Tensor): Predicted labels (B,).
        y_gt (torch.Tensor): Ground truth labels (B,).

    Returns:
        torch.Tensor: Tensor of shape (N,), where N is the number of correct predictions.
                      Values are 1.0 if verified, 0.0 if not.
    """
    with torch.no_grad():
        lb = output_bounds.lower 
        ub = output_bounds.upper

        correct_mask = (y_pred == y_gt)
        if correct_mask.sum() == 0:
            return []

        y_pred = y_pred[correct_mask]
        y_gt = y_gt[correct_mask]
        lb = lb[correct_mask]
        ub = ub[correct_mask]

        batch_size = lb.size(0)
        idx = torch.arange(batch_size, device=lb.device)

        gt_lb = lb[idx, y_gt]

        # Mask out the correct class
        non_gt_mask = torch.ones_like(ub, dtype=torch.bool)
        non_gt_mask[idx, y_gt] = False
        ub_non_gt = ub[non_gt_mask].view(batch_size, -1)

        # Verified if ground truth lower bound is greater than all upper bounds of other classes
        is_verified = (gt_lb.unsqueeze(1) > ub_non_gt).all(dim=1)
       
        return is_verified.float()



    
def check_correct_prediction(module: nn.Module, x: torch.Tensor, y_gt: torch.Tensor) -> Tuple[bool,torch.Tensor]:
    """
    Checks whether a given neural network module correctly classifies an input tensor.
    Args:
        module (nn.Module): The neural network module to evaluate.
        x (torch.Tensor): The input tensor to be classified by the module.
        y_gt (torch.Tensor): The ground truth tensor representing the correct classification.
    Returns: Tuple
        bool: True if the module's prediction matches the ground truth, False otherwise.
        torch.tensor: 
    """
    y_pred = module(x)
    y_pred = torch.argmax(y_pred, dim=-1)
    correctly_classified = y_pred == y_gt
    
    return correctly_classified, y_pred
    
def add_salt_and_pepper(x: torch.Tensor, amount: float = 0.02, salt_vs_pepper: float = 0.5) -> torch.Tensor:
    """
    Adds salt and pepper noise to a tensor, supporting flattened or image-shaped tensors.
    Automatically adapts to the value range of the input tensor.

    Args:
        x (torch.Tensor): Input tensor (flattened or shaped), any value range.
        amount (float): Proportion of values to alter.
        salt_vs_pepper (float): Proportion of salt noise vs pepper.

    Returns:
        torch.Tensor: Noisy tensor.
    """
    noisy = x.clone()
    noisy = squeeze_batch_dim(noisy)
    min_val = noisy.min()
    max_val = noisy.max()

    if noisy.dim() == 1:
        num_elements = noisy.numel()
        num_salt = int(amount * num_elements * salt_vs_pepper)
        num_pepper = int(amount * num_elements * (1.0 - salt_vs_pepper))

        # Salt
        indices = torch.randint(0, num_elements, (num_salt,))
        noisy[indices] = max_val

        # Pepper
        indices = torch.randint(0, num_elements, (num_pepper,))
        noisy[indices] = min_val

    else:
        if noisy.dim() == 3:
            noisy = noisy.unsqueeze(0)
        N, C, H, W = noisy.shape
        num_pixels = H * W
        num_salt = int(amount * num_pixels * salt_vs_pepper)
        num_pepper = int(amount * num_pixels * (1.0 - salt_vs_pepper))

        for img in noisy:
            coords = [torch.randint(0, H, (num_salt,)), torch.randint(0, W, (num_salt,))]
            img[:, coords[0], coords[1]] = max_val

            coords = [torch.randint(0, H, (num_pepper,)), torch.randint(0, W, (num_pepper,))]
            img[:, coords[0], coords[1]] = min_val

        if x.dim() == 3:
            noisy = noisy.squeeze(0)

    return noisy

def save_deteriotated_image(x: torch.Tensor, flatten: bool, folder: str, 
                            dataset: str, path_suffix: str = None) -> None:
    """
    Save a deteriorated image from a tensor to disk.

    Args:
        x (torch.Tensor): Image tensor (either flattened or in CHW format).
        flatten (bool): Whether the tensor is flattened (e.g., from a dataset like MNIST).
        folder (str): Path to the directory where the deteriorated image will be saved.
        dataset (str): Name of the dataset (needed if flatten=True).
        path_suffix (Optional, str): Name suffix.
    """
    x = x.clone().detach()
    x = squeeze_batch_dim(x)

    if flatten:
        if "MNIST" in dataset:
            x = x.view(1, 28, 28)
        elif "CIFAR10" in dataset or "SVHN" in dataset:
            x = x.view(3, 32, 32)
        else:
            raise ValueError(f"Flattened image saving not implemented for dataset: {dataset}")

    if x.max() <= 1.0:
        x = x * 255.0
    x = x.to(torch.uint8)

    to_pil = transforms.ToPILImage()
    img = to_pil(x)

    os.makedirs(f"{folder}/deteriorated_images", exist_ok=True)
    if path_suffix is not None:
        img_path = f"{folder}/deteriorated_images/image_{uuid.uuid4().hex[:8]}{path_suffix}.png"
    else:
        img_path = f"{folder}/deteriorated_images/image_{uuid.uuid4().hex[:8]}.png"
    img.save(img_path)

def generate_boundary_points(method: MethodPluginABC, X: torch.Tensor, y: torch.Tensor, 
                             perturbation: float = 1e-2, grad_steps: int = 3) -> torch.Tensor:
    """
    Generates perturbed inputs near the decision boundary by applying
    small gradient-based perturbations over multiple steps.

    Args:
        method (MethodPluginABC): The method plugin used for predictions.
        X (torch.Tensor): The input tensor for which boundary points are to be generated.
        y (torch.Tensor): The ground truth labels corresponding to the input tensor.
        perturbation (float, optional): The magnitude of the perturbation to be applied per step. Defaults to 1e-2.
        grad_steps (int, optional): Number of gradient ascent steps to apply. Defaults to 3.

    Returns:
        torch.Tensor: A tensor containing the perturbed inputs near the decision boundary.
    """
    X_adv = X.clone().detach()

    for _ in range(grad_steps):
        X_adv.requires_grad_(True)
        y_pred = method.module(X_adv)
        loss = torch.nn.functional.cross_entropy(y_pred, y)
        loss.backward()

        with torch.no_grad():
            grad = X_adv.grad
            X_adv = X_adv + perturbation * torch.sign(grad)
            X_adv = torch.clamp(X_adv, min=0.0, max=1.0)
        
        X_adv = X_adv.detach()

    return X_adv

def get_eps(config: DictConfig, eps: torch.Tensor) -> torch.Tensor:
    """
    Adjusts the epsilon tensor by dividing it by the dataset's standard deviation
    if the standard deviation is defined in the dataset configuration.

    Args:
        config (DictConfig): The configuration object containing dataset information.
            It is expected to have a nested structure where `config.dataset.dataset.std`
            holds the standard deviation values as a list or tensor.
        eps (torch.Tensor): The epsilon tensor to be adjusted. Typically represents
            perturbation bounds in adversarial training or similar contexts.

    Returns:
        torch.Tensor: The adjusted epsilon tensor. If the dataset's standard deviation
            is defined, the epsilon tensor is divided by it. Otherwise, the original
            epsilon tensor is returned unchanged.
    """
    if hasattr(config.dataset.dataset, "std"):
        std = config.dataset.dataset.std
        std = torch.tensor(std, device=eps.device).view(1, 3, 1, 1)
        eps = eps / std
    return eps
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms

import os
import uuid
from typing import Union

from method.interval_arithmetic import Interval
from dataset.mnist import MNIST
from dataset.cifar10 import CIFAR10
from dataset.svhn import SVHN
from method.method_plugin_abc import MethodPluginABC

from omegaconf import DictConfig
from hydra.utils import instantiate


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

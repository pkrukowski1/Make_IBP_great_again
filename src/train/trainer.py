import torch
import logging
from typing import Tuple

from method.interval_arithmetic import Interval
from method.affine_arithmetic import AffineNN
from method.method_plugin_abc import MethodPluginABC

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Trainer:
    """
    Trainer class for verifiable training using interval bound propagation (e.g., Affine Arithmetic),
    with epsilon and kappa warm-up scheduling.
    """

    def __init__(
        self,
        method: MethodPluginABC,
        start_epoch: int = 0,
        end_epoch: int = 100,
    ) -> None:
        """
        Initializes the Trainer.

        Args:
            used_method (MethodPluginABC): Bound propagation method to use.
            start_epoch (int): Epoch to begin increasing epsilon and decreasing kappa.
            end_epoch (int): Epoch to reach full epsilon and zero kappa.
        """
        super().__init__()
        self.epsilon_schedule = method.plugins[0].epsilon
        self.current_epsilon = 0.0

        self.kappa_schedule = True
        self.current_kappa = 1.0

        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.current_epoch = 0

        self.method = method.plugins[0]
      
        log.info(f"Trainer initialized with epsilon = {self.epsilon_schedule}]")

    def set_epoch(self, epoch: int) -> None:
        """
        Updates epsilon and kappa according to warm-up schedule.

        Args:
            epoch (int): Current epoch.
        """
        self.current_epoch = epoch

        if epoch < self.start_epoch:
            self.current_epsilon = 0.0
            self.current_kappa = 1.0
        elif epoch >= self.end_epoch:
            self.current_epsilon = self.epsilon_schedule
            self.current_kappa = 0.0
        else:
            progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            self.current_epsilon = progress * self.epsilon_schedule
            self.current_kappa = 1.0 - progress

        self.method.epsilon = self.current_epsilon

        log.info(f"Epoch {epoch}: epsilon = {self.current_epsilon:.5f}, kappa = {self.current_kappa:.5f}")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Interval]:
        """
        Performs a forward pass and computes combined natural + robust loss.

        Args:
            x (torch.Tensor): Input batch.
            y (torch.Tensor): Ground truth labels.

        Returns:
            Tuple[torch.Tensor, Interval]: Total loss, and interval output bounds.
        """
        # Eps will be multiplied by a proper value inside the `self.method.forward` method.
        eps = torch.ones_like(x, device=x.device)

        # Interval bounds (robust)
        out_bounds = self.method.forward(x, y, eps)

        # Natural output (standard logits)
        logits = self.method.module(x)

        # Combined loss
        loss = self.calculate_loss(logits, out_bounds, y)

        return loss, out_bounds

    def calculate_loss(self, logits: torch.Tensor, bounds: Interval, y: torch.Tensor) -> torch.Tensor:
        """
        Combines standard and robust loss using scheduled kappa.

        Args:
            logits (torch.Tensor): Natural output logits.
            bounds (Interval): Interval bounds from robust forward.
            y (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Scalar total loss.
        """
        # Natural CE loss
        natural_loss = torch.nn.functional.cross_entropy(logits, y)

        # Robust hinge-style loss
        lower = bounds.lower
        upper = bounds.upper

        batch_size = lower.size(0)
        num_classes = lower.size(1)

        true_class_lower = lower[torch.arange(batch_size), y]

        mask = torch.ones_like(upper, dtype=torch.bool)
        mask[torch.arange(batch_size), y] = False
        worst_other_upper = upper[mask].view(batch_size, num_classes - 1).max(dim=1)[0]

        margins = true_class_lower - worst_other_upper
        margin_logits = torch.stack([margins, torch.zeros_like(margins)], dim=1)
        targets = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        robust_loss = torch.nn.functional.cross_entropy(margin_logits, targets)

        # Combine
        loss = self.current_kappa * natural_loss + (1.0 - self.current_kappa) * robust_loss

        return loss

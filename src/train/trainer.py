import torch
import logging
from typing import Tuple

from method.interval_arithmetic import Interval
from method.method_plugin_abc import MethodPluginABC

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Trainer:
    """
    Trainer class responsible for scheduling epsilon and kappa during training,
    and computing loss for robust training using interval bounds.

    Epsilon is linearly increased and kappa linearly decreased per batch
    after the warmup period, over a fixed number of scheduling epochs.
    """

    def __init__(
        self,
        method: MethodPluginABC,
        start_epoch: int = 0,
        end_epoch: int = 100,
        schedule_epochs: int = 10,
        num_batches_per_epoch: int = 100,
    ) -> None:
        """
        Initializes the Trainer.

        Args:
            method (MethodPluginABC): The training method (plugin) with an interval bound forward pass.
            start_epoch (int): The first epoch (inclusive) of the warmup phase.
            end_epoch (int): The last epoch (exclusive) of the warmup phase.
            schedule_epochs (int): Number of epochs to linearly schedule epsilon and kappa after warmup.
            num_batches_per_epoch (int): Number of batches per training epoch.
        """
        super().__init__()
        self.method = method.plugins[0]
        self.epsilon_train = self.method.epsilon
        self.current_epsilon = 0.0
        self.current_kappa = 1.0

        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.schedule_epochs = schedule_epochs
        self.num_batches_per_epoch = num_batches_per_epoch

        self.schedule_total_steps = schedule_epochs * num_batches_per_epoch
        self.schedule_step = 0
        self.current_epoch = 0

        log.info(f"Trainer initialized with epsilon schedule = {self.epsilon_train}, "
                 f"schedule steps = {self.schedule_total_steps}")

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the current epoch, resetting the batch step counter and updating initial
        epsilon and kappa values based on the epoch.

        Args:
            epoch (int): The epoch number to set.
        """
        self.current_epoch = epoch
        self.schedule_step = 0

        if epoch < self.end_epoch:
            self.current_epsilon = 0.0
            self.current_kappa = 1.0
        elif self.schedule_total_steps == 0:
            self.current_epsilon = self.epsilon_train
            self.current_kappa = 0.0

        self.method.epsilon = self.current_epsilon

        log.info(f"[Epoch {epoch}] Initialized epoch. epsilon = {self.current_epsilon:.5f}, "
                 f"kappa = {self.current_kappa:.5f}")

    def step_batch_schedule(self):
        """
        Updates epsilon and kappa linearly per batch during the scheduling phase.
        If warmup hasn't ended or schedule is complete, values remain static.
        """
        if self.current_epoch < self.end_epoch:
            return  # warmup phase

        if self.schedule_step >= self.schedule_total_steps:
            self.current_epsilon = self.epsilon_train
            self.current_kappa = 0.0
            return

        progress = self.schedule_step / self.schedule_total_steps
        self.current_epsilon = progress * self.epsilon_train
        self.current_kappa = max(1.0 - progress, 0.0)
        self.schedule_step += 1
        self.method.epsilon = self.current_epsilon

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Interval]:
        """
        Performs a forward pass using the current epsilon, computes robust interval bounds,
        and returns the loss and bounds.

        Args:
            x (torch.Tensor): Input batch.
            y (torch.Tensor): Ground truth labels.

        Returns:
            Tuple[torch.Tensor, Interval]: Loss and interval bounds for the batch.
        """
        if self.method.module.training:
            self.step_batch_schedule()
        eps = torch.ones_like(x, device=x.device)
        out_bounds = self.method.forward(x, y, eps)
        logits = self.method.module(x)
        loss = self.calculate_loss(logits, out_bounds, y)
        return loss, out_bounds

    def calculate_loss(self, logits: torch.Tensor, bounds: Interval, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the mixed loss from natural and robust cross-entropy losses.

        Args:
            logits (torch.Tensor): Model outputs.
            bounds (Interval): Interval bounds of the outputs.
            y (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Combined loss.
        """
        natural_loss = torch.nn.functional.cross_entropy(logits, y)

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
        loss = self.current_kappa * natural_loss + (1.0 - self.current_kappa) * robust_loss

        return loss

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

import torch

import logging

from method.method_plugin_abc import MethodPluginABC
from method.interval_arithmetic import Interval

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class AlphaCROWN(MethodPluginABC):
    """
    AlphaCROWN is a method plugin for performing certified robustness verification using the 
    CROWN-Optimized (alpha-CROWN) approach. It leverages linear relaxation-based bounds 
    to compute certified intervals for neural network outputs under adversarial perturbations.

    Attributes:
       epsilon (float): The perturbation scaling factor.
        gradient_iter (int): The number of iterations for optimizing the bounds during 
            the verification process.
        lr (float): The learning rate for optimizing the alpha parameters in the 
            CROWN-Optimized method.

        lirpa_model (BoundedModule or None): A model wrapped with the LiRPA (Linear 
            Relaxation-based Perturbation Analysis) library for certified robustness 
            verification. Initialized as None and set during the first forward pass.
        _method (str): The method name used for bound computation, set to 
            "CROWN-Optimized".
        _norm (float): The norm type used for perturbation analysis, set to infinity norm.
        _ptb (PerturbationLpNorm): The perturbation object defining the norm and epsilon 
            for adversarial perturbations.

    Methods:
        __init__(epsilon: float, gradient_iter: int, lr: float) -> None:
            Initializes the AlphaCROWN plugin with the specified parameters.

        forward(x: torch.Tensor, y: torch.Tensor, eps: torch.Tensor) -> Interval:
            Computes the certified lower and upper bounds for the neural network's output 
            given an input tensor `x`, a target tensor `y`, and a scaling factor `eps`.

            Args:
                x (torch.Tensor): The input tensor to the neural network.
                y (torch.Tensor): The target tensor (not used in this method).
                eps (torch.Tensor): A scaling factor for the perturbation radii.

            Returns:
                Interval: An object containing the lower and upper bounds for the 
                neural network's output under the specified perturbation budget.
    """
    def __init__(self, epsilon: float, gradient_iter: int, lr: float):
        super().__init__()

        self.epsilon = epsilon

        self.lirpa_model = None
        self.gradient_iter = gradient_iter
        self.lr = lr

        self._method = "CROWN-Optimized"
        self._norm = float("inf")
        self._ptb = PerturbationLpNorm(norm = self._norm, eps = epsilon)

        log.info(f"AlphaCROWN plugin initialized for epsilon={self.epsilon}")

    def forward(self, x: torch.Tensor, y: torch.Tensor, eps: torch.Tensor) -> Interval:
        """
        Perform a forward pass through the model with Alpha-CROWN.

        Args:
            x (torch.Tensor): The input tensor to the model.
            y (torch.Tensor): The target tensor (not used in this method).
            eps (torch.Tensor): A tensor representing the perturbation range. The final radii
                for the perturbation are computed as `self.epsilon * eps`.

        Returns:
            Interval: An interval object containing the lower and upper bounds 
                      of the output after applying the specified perturbation 
                      and computing bounds using the LiRPA model.

        Notes:
            - The method initializes a perturbation object (`PerturbationLpNorm`) 
              with the specified norm and a scaled epsilon (`self.epsilon * eps`).
            - If the LiRPA model (`lirpa_model`) is not already initialized, it 
              creates a `BoundedModule` for the given model and sets optimization 
              options for bound computation.
            - The bounds are computed using the specified method (`self._method`).
        """
        epsilon = eps * self.epsilon
        ptb = PerturbationLpNorm(norm = self._norm, eps = epsilon)
        x = BoundedTensor(x, ptb)

        if self.lirpa_model is None:
            self.lirpa_model = BoundedModule(self.module, torch.empty_like(x), device=x.device)
            self.lirpa_model.set_bound_opts({'optimize_bound_args': {
                'iteration': self.gradient_iter, 'lr_alpha': self.lr
                }})

        lb, ub = self.lirpa_model.compute_bounds(x=(x,), method=self._method)

        return Interval(lb, ub)
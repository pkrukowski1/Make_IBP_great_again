from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

import torch

import logging

from method.method_plugin_abc import MethodPluginABC
from method.interval_arithmetic import Interval

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class CROWN(MethodPluginABC):
    """
    CROWN is a method plugin for interval bound propagation (IBP) that computes
    certified bounds for neural network outputs under input perturbations.

    Attributes:
        epsilon (float): The perturbation scaling factor.
        lirpa_model (BoundedModule or None): The model wrapped with bounds
            computation capabilities. Initialized lazily during the first forward pass.
        _method (str): The method used for bounds computation, set to "CROWN".
        _norm (float): The norm used for perturbation, set to infinity norm.
        _ptb (PerturbationLpNorm): The perturbation object initialized with the
            specified norm and epsilon.

    Methods:
        __init__(epsilon: float):
            Initializes the CROWN plugin with the specified perturbation radius.

        forward(x: torch.Tensor, y: torch.Tensor, eps: torch.Tensor) -> Interval:
            Computes the lower and upper bounds for the neural network's output
            under input perturbations. If the lirpa_model is not initialized, it
            wraps the module with a BoundedModule for bounds computation.

            Args:
                x (torch.Tensor): The input tensor to the neural network.
                y (torch.Tensor): An additional input tensor (not used in this implementation).
                eps (torch.Tensor): A scaling factor for the perturbation radius.

            Returns:
                Interval: An object containing the lower and upper bounds for the
                neural network's output.
    """
   
    def __init__(self, epsilon: float):
        super().__init__()

        self.epsilon = epsilon

        self.lirpa_model = None

        self._method = "CROWN"
        self._norm = float("inf")
        self._ptb = PerturbationLpNorm(norm = self._norm, eps = epsilon)

        log.info(f"CROWN plugin initialized for epsilon={self.epsilon}")

    def forward(self, x: torch.Tensor, y: torch.Tensor, eps: torch.Tensor) -> Interval:
        """
        Perform a forward pass through the model and compute the interval bounds.
        Args:
            x (torch.Tensor): The input tensor to the model.
            y (torch.Tensor): An additional input tensor (not used in this implementation).
            eps (torch.Tensor): A tensor representing the perturbation range. The final radii
                for the perturbation are computed as `self.epsilon * eps`.

        Interval: An object containing the lower bound (`lb`) and upper bound (`ub`)

        - This method applies a perturbation to the input tensor `x` using the specified
            Lp-norm and a scaled epsilon value (`eps * self.epsilon`).
        - The `PerturbationLpNorm` is used to define the perturbation applied to the input.
        """
        epsilon = eps * self.epsilon
        ptb = PerturbationLpNorm(norm = self._norm, eps = epsilon)
        x = BoundedTensor(x, ptb)

        if self.lirpa_model is None:
            self.lirpa_model = BoundedModule(self.module, torch.empty_like(x), device=x.device)

        lb, ub = self.lirpa_model.compute_bounds(x=(x,), method=self._method)

        return Interval(lb, ub)
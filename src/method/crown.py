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
        epsilon (float): The perturbation radius for the input. Determines the
            maximum allowable deviation for input perturbations.
        lirpa_model (BoundedModule or None): The model wrapped with bounds
            computation capabilities. Initialized lazily during the first forward pass.
        _method (str): The method used for bounds computation, set to "CROWN".
        _norm (float): The norm used for perturbation, set to infinity norm.
        _ptb (PerturbationLpNorm): The perturbation object initialized with the
            specified norm and epsilon.
    Methods:
        __init__(epsilon: float):
            Initializes the CROWN plugin with the specified perturbation radius.
        forward(x: torch.Tensor, y: torch.Tensor) -> Interval:
            Computes the lower and upper bounds for the neural network's output
            under input perturbations. If the lirpa_model is not initialized, it
            wraps the module with a BoundedModule for bounds computation.
            Args:
                x (torch.Tensor): The input tensor to the neural network.
                y (torch.Tensor): The target tensor (not used in this implementation).
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Interval:
        """
        Perform a forward pass through the model and compute the interval bounds.
        Args:
            x (torch.Tensor): The input tensor to the model.
            y (torch.Tensor): An additional input tensor (not used in this implementation).
        Returns:
            Interval: An interval object containing the lower bound (`lb`) and upper bound (`ub`)
                      of the computed bounds for the input tensor.
        Notes:
            - The method applies a perturbation to the input tensor `x` using the specified
              Lp-norm and epsilon value.
            - If the `lirpa_model` is not initialized, it creates a `BoundedModule` using
              the provided `module` and the shape of the input tensor.
            - The bounds are computed using the specified method (`self._method`).
        """
        
        ptb = PerturbationLpNorm(norm = self._norm, eps = self.epsilon)
        x = BoundedTensor(x, ptb)

        if self.lirpa_model is None:
            self.lirpa_model = BoundedModule(self.module, torch.empty_like(x), device=x.device)

        lb, ub = self.lirpa_model.compute_bounds(x=(x,), method=self._method)

        return Interval(lb, ub)
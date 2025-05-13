import torch

import logging

from method.method_plugin_abc import MethodPluginABC
from method.interval_arithmetic import Interval

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class LowerBound(MethodPluginABC):
    """
    LowerBound is a plugin class that computes the lower and upper bounds of a model's output
    for a given input tensor within a specified epsilon neighborhood. It uses random sampling
    to estimate the bounds.
    Attributes:
        epsilon (float): The radius of the neighborhood around the input tensor for which
            the bounds are computed.
        num_points (int): The number of random points to sample within the epsilon neighborhood.
    Methods:
        __init__(epsilon: float, num_points: int = 1000):
            Initializes the LowerBound plugin with the specified input dimensionality,
            epsilon radius, and number of sampling points.
        forward(x: torch.Tensor, y: torch.Tensor) -> Interval:
            Computes the lower and upper bounds of the model's output for the given input tensor
            `x` within the epsilon neighborhood. Returns an Interval object containing the
            minimum and maximum bounds.
    """

    def __init__(self, epsilon: float, num_points: int = 1000) -> None:
        """
        Initializes the lower bound method with the given parameters.

        Args:
            epsilon (float): The epsilon value used for the bound calculation.
            num_points (int, optional): The number of points to use in the calculation. Defaults to 1000.

        Returns:
            None
        """
        super().__init__()

        self.epsilon = epsilon
        self.num_points = num_points

        log.info(f"Ideal bound plugin initialized for epsilon={self.epsilon}")
    

    def forward(self, x: torch.Tensor, y: torch.Tensor, eps: torch.Tensor) -> Interval:

        """
        Performs a forward pass to compute the interval bounds for the given input tensor `x`.

        Args:
            x (torch.Tensor): The input tensor for which the interval bounds are to be computed.
            y (torch.Tensor): An additional input tensor (not used in this implementation).
            eps (torch.Tensor): A tensor representing the perturbation range for generating random noise 
                within the interval. It defines the maximum deviation allowed for the input tensor `x`.

        Returns:
            Interval: An object representing the computed lower and upper bounds for the output.

        Notes:
            - The method uses a random sampling approach to generate points within the interval
              defined by `x` and `epsilon`.
            - The module is evaluated in `eval` mode, and gradients are not computed during this process.
            - The lower and upper bounds of the output are determined by the minimum and maximum
              values of the module's outputs over the sampled points.
        """
        self.module.eval()
        B = x.size(0)
        shape = x.shape[1:]

        # Expand each input to (num_points) versions
        x_expanded = x.unsqueeze(1).expand(B, self.num_points, *shape)  # (B, N, ...)
        noise = (torch.rand_like(x_expanded) * 2 - 1) * self.epsilon * eps
        perturbed = x_expanded + noise
        perturbed = perturbed.view(-1, *shape)  # (B * N, ...)

        with torch.no_grad():
            outputs = self.module(perturbed)  # (B * N, output_dim)

        outputs = outputs.view(B, self.num_points, -1)  # (B, N, output_dim)
        min_bounds = outputs.min(dim=1).values  # (B, output_dim)
        max_bounds = outputs.max(dim=1).values  # (B, output_dim)

        return Interval(lower=min_bounds, upper=max_bounds)
    
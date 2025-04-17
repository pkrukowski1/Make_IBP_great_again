import logging
from typing import Optional

from torch import nn
import torch

from method.method_plugin_abc import MethodPluginABC
from method.interval_arithmetic import Interval

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class Composer:
    """
    Composer is a class that facilitates the composition of a neural network module with a list of plugins. 
    Each plugin is expected to implement the `MethodPluginABC` interface and can modify the behavior of the 
    module during the forward pass.
    Attributes:
        module (nn.Module): The neural network module to which plugins are applied.
        epsilon (float): The interval radii.
        plugins (Optional[list[MethodPluginABC]]): A list of plugins that modify the behavior of the module. 
            Defaults to an empty list.
    Methods:
        __init__(module: nn.Module, plugins: Optional[list[MethodPluginABC]] = []):
            Initializes the Composer with a module and an optional list of plugins. 
            Each plugin is associated with the module.
        forward(x: torch.Tensor, y: torch.Tensor) -> Interval:
            Executes the forward pass through the module and applies each plugin in sequence.
    """
    def __init__(self, 
        module: nn.Module,
        epsilon: float,
        plugins: Optional[list[MethodPluginABC]]=[]
    ) -> None:

        self.module = module
        self.plugins = plugins
        self.epsilon = epsilon

        for plugin in self.plugins:
            plugin.set_module(self.module)
            log.info(f'Plugin {plugin.__class__.__name__} added to composer')


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Interval:
        """
        Forward pass through the module and plugins.
        
        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
        
        Returns:
            Interval: Output interval bounds after passing through the module and plugins.
        """
        for plugin in self.plugins:
            x = plugin.forward(x, y)
        
        return x
        


   
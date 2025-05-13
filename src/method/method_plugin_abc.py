from abc import ABCMeta, abstractmethod

from torch import nn


class MethodPluginABC(metaclass=ABCMeta):    
    """
    An abstract base class for method plugins that operate on neural network modules.
    This class defines the interface for plugins that can be attached to a neural 
    network module and perform specific operations during the forward pass.
    Methods:
        set_module(module: nn.Module):
            Sets the neural network module for the plugin to operate on.
        forward(x,y,eps):
            Abstract method that defines the internal forward pass logic. 
            Must be implemented by subclasses.
    Attributes:
        module (nn.Module):
            The neural network module associated with the plugin.
    """
    def set_module(self, module: nn.Module) -> None:
        """
        Set the module for the plugin.
        
        Args:
            module(nn.Module): The model to be set.
        """

        self.module = module

    @abstractmethod
    def forward(self, x, y, eps) -> None:
        """
        Internal forward pass.
        """
        pass
from torch import nn
import torch

import logging

from .utils import load_conv_model

log = logging.getLogger(__name__)


class ConvSmallMNIST(nn.Module):
    """
    A convolutional neural network model designed for small-scale image processing tasks.
    The model consists of a feature extractor using convolutional layers followed by a classifier.
    Attributes:
        in_channels (int): Number of input channels for the convolutional layers.
        dim_out (int): Dimension of the output layer (number of classes or regression targets).
        model_path (str, optional): Path to a pre-trained model to load. Defaults to None.
    Methods:
        __init__(dim_out: int, model_path: str = None):
            Initializes the ConvSmall model.
        _build() -> nn.Sequential:
            Builds the architecture of the ConvSmall model.
        forward(x: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the ConvSmall model.
    """
        
    def __init__(self, dim_out: int, model_path: str = None) -> None:
        """
        Initializes the ConvSmall model.
        Args:
            dim_out (int): Dimension of the output layer (number of classes or regression targets).
            model_path (str, optional): Path to a pre-trained model to load. If provided, the model
                                         weights will be loaded from this path. Defaults to None.
        """
        super().__init__()

        self.in_channels = 1
        self.input_height = 28
        self.input_width = 28
        self.dim_out = dim_out

        self.model = self._build()

        if model_path is not None:
            load_conv_model(self.model, model_path)
            log.info(f"Model loaded from {model_path}")

    def _build(self) -> nn.Sequential:
        """
        Builds the architecture of the ConvSmall model, including the feature extractor and classifier.
        Returns:
            nn.Sequential: A sequential container of the feature extractor and classifier.
        """
        feature_extractor = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, (4,4), 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4,4), 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Dynamically determine the number of features after flattening
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, self.input_height, self.input_width)
            hidden_units = feature_extractor(dummy_input).shape[1]

        classifier = nn.Sequential(
            nn.Linear(hidden_units, 100),
            nn.ReLU(),
            nn.Linear(100, self.dim_out),
        )

        return nn.Sequential(feature_extractor, classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the ConvSmall model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, dim_out).
        """
        return self.model(x)

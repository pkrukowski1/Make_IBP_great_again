import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

log = logging.getLogger(__name__)

from .utils import load_model

class BasicBlock(nn.Module):
    """
    A basic building block for a ResNet-like neural network.
    This block consists of two convolutional layers, optional batch normalization, 
    and a shortcut connection. It supports different kernel sizes and strides.
    Attributes:
        expansion (int): Expansion factor for the number of output channels.
        conv1 (nn.Conv2d): The first convolutional layer.
        bn1 (nn.BatchNorm2d, optional): Batch normalization for the first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        bn2 (nn.BatchNorm2d, optional): Batch normalization for the second convolutional layer.
        shortcut (nn.Sequential): The shortcut connection, which may include a convolutional 
            layer and optional batch normalization.
    Args:
        in_planes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride for the first convolutional layer. Default is 1.
        bn (bool, optional): Whether to use batch normalization. Default is True.
        kernel (int, optional): Kernel size for the convolutional layers. Supported values are 
            1, 2, or 3. Default is 3.
    Methods:
        forward(x):
            Defines the forward pass of the block. Applies two convolutional layers with 
            optional batch normalization and ReLU activation, adds the shortcut connection, 
            and applies a final ReLU activation.
    Raises:
        SystemExit: If an unsupported kernel size is provided.
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1, bn: bool=True, kernel: int=3) -> None:
        super(BasicBlock, self).__init__()
        self.bn = bn
        if kernel == 3:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=(not self.bn))
        elif kernel == 2:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=2, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=2,
                                   stride=1, padding=0, bias=(not self.bn))
        elif kernel == 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                                   stride=1, padding=0, bias=(not self.bn))
        else:
            exit("kernel not supported!")

        if self.bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if self.bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=(not self.bn)),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=(not self.bn)),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.
        Args:
            x (torch.Tensor): Input tensor to the network.
        Returns:
            torch.Tensor: Output tensor after applying the forward pass, which includes
            convolutional layers, optional batch normalization, ReLU activations, and
            a residual connection.
        """

        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet5(nn.Module):
    """
    A custom implementation of a ResNet-like neural network with configurable layers and options.
    Args:
        block (nn.Module): The building block for the network (e.g., BasicBlock or Bottleneck).
        num_blocks (int, optional): Number of blocks in the first layer. Default is 2.
        num_classes (int, optional): Number of output classes for classification. Default is 10.
        in_planes (int, optional): Number of input channels for the first convolutional layer. Default is 64.
        bn (bool, optional): Whether to use Batch Normalization. Default is True.
        last_layer (str, optional): Type of the last layer. Options are "avg" for average pooling 
                                    or "dense" for fully connected layers. Default is "avg".
    Attributes:
        in_planes (int): Tracks the number of input channels for the current layer.
        bn (bool): Indicates whether Batch Normalization is used.
        last_layer (str): Specifies the type of the last layer.
        conv1 (nn.Conv2d): The first convolutional layer.
        bn1 (nn.BatchNorm2d, optional): Batch Normalization layer for the first convolutional layer.
        layer1 (nn.Sequential): The first layer of blocks created using `_make_layer`.
        avg2d (nn.AvgPool2d, optional): Average pooling layer used when `last_layer` is "avg".
        linear (nn.Linear, optional): Fully connected layer used when `last_layer` is "avg".
        linear1 (nn.Linear, optional): First fully connected layer used when `last_layer` is "dense".
        linear2 (nn.Linear, optional): Second fully connected layer used when `last_layer` is "dense".
    Methods:
        _make_layer(block, planes, num_blocks, stride, bn, kernel):
            Creates a sequential layer of blocks with the specified parameters.
        forward(x):
            Defines the forward pass of the network.
    Raises:
        SystemExit: If an unsupported `last_layer` type is provided.
    """

    def __init__(self, 
                 block: nn.Module, 
                 num_blocks: int=2, 
                 num_classes: int=10, 
                 in_planes: int=64, 
                 bn: bool=True, 
                 last_layer: str="avg",
                 model_path: str=None) -> None:
        """
        Initializes the ResNet5 model.
        Args:
            block (nn.Module): The building block for the ResNet5 layers.
            num_blocks (int, optional): Number of blocks in each layer. Default is 2.
            num_classes (int, optional): Number of output classes for classification. Default is 10.
            in_planes (int, optional): Number of input planes for the first convolutional layer. Default is 64.
            bn (bool, optional): Whether to use Batch Normalization. Default is True.
            last_layer (str, optional): Type of the last layer. Options are:
                - "avg": Uses an average pooling layer followed by a fully connected layer.
                - "dense": Uses two fully connected layers.
                Default is "avg".
            model_path (str, optional): Path to a pre-trained model to load. If provided, the model weights 
                        will be loaded from this path. Default is None.
        Raises:
            SystemExit: If an unsupported `last_layer` type is provided.
        """

        super(ResNet5, self).__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.last_layer = last_layer
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=2, padding=1, bias=not self.bn)
        if self.bn: self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        if self.last_layer == "avg":
            self.avg2d = nn.AvgPool2d(4)
            self.linear = nn.Linear(in_planes * 8 * block.expansion, num_classes)
        elif self.last_layer == "dense":
            self.linear1 = nn.Linear(in_planes * 8 * block.expansion * 16, 100)
            self.linear2 = nn.Linear(100, num_classes)
        else:
            exit("last_layer type not supported!")

        if model_path is not None:
            load_model(self.model, model_path)
            log.info(f"Model loaded from {model_path}")

    def _make_layer(self, 
                    block: nn.Module, 
                    planes: int, 
                    num_blocks: int, 
                    stride: int, 
                    bn: bool, 
                    kernel: int
                    ) -> nn.Sequential:
        """
        Constructs a sequential layer consisting of multiple blocks.
        Args:
            block (nn.Module): The block class to be used for constructing the layer.
            planes (int): The number of output feature maps for each block.
            num_blocks (int): The number of blocks to include in the layer.
            stride (int): The stride to be used for the first block in the layer.
            bn (bool): A flag indicating whether batch normalization should be applied.
            kernel (int): The kernel size to be used in the convolutional layers.
        Returns:
            nn.Sequential: A sequential container of the constructed blocks.
        """

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.
        Args:
            x (torch.Tensor): Input tensor to the network.
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        Behavior:
            - If batch normalization (`self.bn`) is enabled, applies a convolution
              followed by batch normalization and ReLU activation. Otherwise, applies
              only convolution followed by ReLU activation.
            - Passes the result through `self.layer1`.
            - Depending on the value of `self.last_layer`:
                - If "avg", applies 2D average pooling (`self.avg2d`), reshapes the tensor,
                  and passes it through a fully connected layer (`self.linear`).
                - If "dense", flattens the tensor, applies a ReLU activation followed by
                  a fully connected layer (`self.linear1`), and then another fully connected
                  layer (`self.linear2`).
        """

        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        if self.last_layer == "avg":
            out = self.avg2d(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif self.last_layer == "dense":
            out = torch.flatten(out, 1)
            out = F.relu(self.linear1(out))
            out = self.linear2(out)
        return out
    
def resnet2b() -> ResNet5:
    """
    Constructs a ResNet5 model with specific configurations.
    This function creates a ResNet5 model using the BasicBlock architecture with 
    the following parameters:
    - Number of blocks: 2
    - Initial number of input planes: 8
    - Batch normalization: Disabled
    - Last layer type: Dense
    Returns:
        ResNet5: An instance of the ResNet5 model configured with the specified parameters.
    """

    return ResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="dense")
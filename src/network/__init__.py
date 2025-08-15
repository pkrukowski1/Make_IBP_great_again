# Pretrained
from .mlp import MLP
from .conv_small_mnist import ConvSmallMNIST
from .conv_big_mnist import ConvBigMNIST
from .conv_small_cifar import ConvSmallCIFAR
from .conv_big_cifar import ConvBigCIFAR
from .sdp_model_loader import SDPModelLoader

# To train from scratch
from .conv4layers import Conv4Layers
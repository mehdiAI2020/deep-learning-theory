"""Deep Learning Theory from Scratch - Core concepts with mathematical foundations."""

from .backprop import Tensor, scalar_to_tensor, backward
from .optimizers import SGD, Momentum, Adam
from .normalization import BatchNorm1d, LayerNorm

__all__ = [
    "Tensor",
    "scalar_to_tensor",
    "backward",
    "SGD",
    "Momentum",
    "Adam",
    "BatchNorm1d",
    "LayerNorm",
]
__version__ = "0.1.0"

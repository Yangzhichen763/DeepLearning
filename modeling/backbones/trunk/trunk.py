import torch
import torch.nn as nn

from abc import ABCMeta, abstractmethod


class Trunk(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for backbones.
    """

    def __init__(self):
        super(Trunk, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: input tensor.

        Returns:
            A list of feature maps.
        """
        pass

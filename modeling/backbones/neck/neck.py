import torch
import torch.nn as nn

from abc import ABCMeta, abstractmethod


class Neck(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for backbones.
    """

    def __init__(self):
        super(Neck, self).__init__()

    @abstractmethod
    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            xs (list[torch.Tensor]): list of backbone feature maps.

        Returns:
            list[torch.Tensor]: list of neck feature maps.
        """
        pass

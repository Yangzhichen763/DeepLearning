import torch
import torch.nn as nn

from abc import ABCMeta, abstractmethod


# 参考代码：https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/backbone.py
class Backbone(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for backbones.
    """

    def __init__(self):
        super(Backbone, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

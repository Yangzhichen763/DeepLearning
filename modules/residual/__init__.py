import enum

import modules.residual.ResNet
import modules.residual.ResNeXt
import modules.residual.Res2Net
import modules.residual.Inception


class ShortCutType(enum.Enum):
    IDENTITY = enum.auto()
    CONV1X1 = enum.auto()
    CONV3X3 = enum.auto()

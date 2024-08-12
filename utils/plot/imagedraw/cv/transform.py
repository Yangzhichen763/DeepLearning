import cv2
import enum


class ChannelConvertMethod(enum.Enum):
    FILL_ZERO = enum.auto()
    OPENCV = enum.auto()


def convert_to_channel(src, channels, convert_method=ChannelConvertMethod.FILL_ZERO):
    if src is None:
        raise ValueError("src is None")

    src_channels = src.shape[-1]
    dst_channels = channels
    if src_channels == dst_channels:
        return src.copy()

    if convert_method == ChannelConvertMethod.FILL_ZERO:
        if src_channels == 1 and dst_channels == 3:
            src_empty = cv2.Mat.zeros(src.size(), src.type())
            cv2.merge([src_empty, src_empty, src], src_empty)
            return src_empty.copy()
        if src_channels == 3 and dst_channels == 1:
            src_channel_mats = cv2.split(src)
            return src_channel_mats[0].copy()
    elif convert_method == ChannelConvertMethod.OPENCV:
        if src_channels == 1 and dst_channels == 3:
            return cv2.cvtColor(src=src, code=cv2.COLOR_GRAY2RGB)
        if src_channels == 3 and dst_channels == 1:
            return cv2.cvtColor(src=src, code=cv2.COLOR_RGB2GRAY)

    raise NotImplementedError(f"channels: {src_channels} -> {dst_channels}")


if __name__ == '__main__':
    # 测试
    import matplotlib.pyplot as plt

    _src = cv2.imread('test.jpg')
    _dst = convert_to_channel(_src, 1)
    plt.subplot(121)
    plt.imshow(_src)
    plt.subplot(122)
    plt.imshow(_dst)
    plt.show()

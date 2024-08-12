import sys
sys.path.append(".")
from manager import *
from strategy import *
sys.path.pop()


def process(window_name: str, _strategy: Strategy,
            path: str, image, event: Event, **kwargs):
    """
    对图像 image 进行 strategy 处理，并将处理结果显示在窗口 window_name 中
    """
    _strategy.process(path, image, event, **kwargs)
    event.on_default(path, image, **kwargs)

    Manager.initialize_mouse_callback(window_name)


if __name__ == '__main__':
    import cv2
    # process('window', Crop(), 'path', None, Event())

    img = cv2.imread('cv/test.jpg')
    cv2.imshow('window', img)
    cv2.waitKey(0)
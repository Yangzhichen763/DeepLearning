from typing import Callable

import cv2


class Manager:
    def __init__(self):
        pass

    current_mouse_callback = None

    on_mouse_down: Callable = None
    on_mouse_move: Callable = None
    on_mouse_up: Callable = None
    on_mouse_wheel: Callable[[bool], None] = None  # 传入的参数是 wheel 方向，1 表示滚动到上方，-1 表示滚动到下方
    on_mouse_do: Callable = None     # 传入的参数和 on_mouse 相同，表示鼠标左键按下并移动

    start = None
    end = None
    last = None
    current = None

    # 每次处理一个 Strategy 的时候，需要初始化 mouse callback，使得 cv2 知道哪个图像窗口需要鼠标操作
    @staticmethod
    def initialize_mouse_callback(window_name: str):
        Manager.current_mouse_callback = Manager.on_mouse
        cv2.setMouseCallback(window_name, Manager.on_mouse)

    # 鼠标事件处理中心
    @staticmethod
    def on_mouse(event, x, y, flags, param):
        # 鼠标按下
        if event == cv2.EVENT_LBUTTONDOWN:
            Manager.start = Manager.current = (x, y)

            if Manager.on_mouse_down is not None:
                Manager.on_mouse_down()

            Manager.last = Manager.current
        # 鼠标抬起
        elif event == cv2.EVENT_LBUTTONUP:
            if Manager.start is not None:
                Manager.current = Manager.end = (x, y)

                if Manager.on_mouse_up is not None:
                    Manager.on_mouse_up()

                Manager.start = Manager.end = Manager.last = Manager.current = None
        # 鼠标移动
        elif event == cv2.EVENT_MOUSEMOVE:
            if Manager.start is not None:
                Manager.current = Manager.end = (x, y)

                if Manager.on_mouse_move is not None:
                    Manager.on_mouse_move()

                Manager.last = Manager.current
        # 鼠标滚动
        if event == cv2.EVENT_MOUSEWHEEL:
            if Manager.on_mouse_wheel is not None:
                Manager.on_mouse_wheel(flags)
        # 自定义鼠标事件
        if Manager.on_mouse_do is not None:
            Manager.on_mouse_do(event, x, y, flags, param)

    @staticmethod
    def clear_mouse_callback():
        Manager.on_mouse_down = None
        Manager.on_mouse_move = None
        Manager.on_mouse_up = None
        Manager.on_mouse_wheel = None
        Manager.on_mouse_do = None
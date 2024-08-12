from .mousestrategy import *

import sys
sys.path.append("..")
# noinspection PyUnresolvedReferences
from strategy import Strategy, Event
# noinspection PyUnresolvedReferences
from manager import Manager


from ..strategy import Strategy, Event, StrokeType, ActionState
from ..manager import Manager


class Crop(MouseStrategy):
    def __init__(self, display_window_name, output_window_name):
        super().__init__(display_window_name, output_window_name)
        self.locked = False

    def on_mouse_down(self):
        self.locked = False

    def on_mouse_up(self):
        if self.locked:
            return

        if Manager.start is None or Manager.end is None:
            return

        # 如果始末点靠太近，则无效，直接返回
        if Manager.start == Manager.end:
            return

        # # 绘制剪裁痕迹
        # self.set_parameter(self.display_image, StrokeType.LINE, ActionState.SELECTED)
        # image_transformer.convert_to_channel(image, display_image, 3, ImageTransformer.ChannelConvertMethod.OPENCV)
        # image_drawer.draw_rect(display_image, start, end)
        # reset_parameter()
        #
        # # 显示临时图像（带有剪裁痕迹的原图像）和剪裁图像
        # output_image = image_processor.easy_process(image, lambda p: p.crop(start, end))
        # image_display(output_windows_name, output_image)
        # image_display(display_windows_name, display_image)
        #
        # # 确认是否应用该改变
        # locked = True
        # ensure_result = ensure(path, image, event)
        # if ensure_result == EnsureState.YES:
        #     # 储存剪裁图像，并关闭显示临时图像（带有剪裁痕迹的原图像）和剪裁图像
        #     self.image_display(path, "剪裁", output_image, window_name=output_windows_name)
        #     self.image_close(display_windows_name)
        #     self.image_close(output_windows_name)
        #
        #     # 对剪裁图像进行处理
        #     rect = Rect(min(start.X, end.X), min(start.Y, end.Y), abs(end.X - start.X), abs(end.Y - start.Y))
        #     rect_args = ProcessArgs(rect)
        #     if event.on_up:
        #         event.on_up(path, output_image, rect_args)
        #
        #     # 清理剪裁痕迹
        #     display_image = image.clone()
        # elif ensure_result == EnsureState.NO:
        #     # 关闭显示剪裁图像
        #     self.image_close(output_windows_name)
        #
        #     # 清理剪裁痕迹，并显示原图像
        #     display_image = image.clone()
        #     self.image_display(display_windows_name, display_image)
        #
        # locked = False

    def on_mouse_move(self):
        pass

    def on_mouse_wheel(self):
        pass

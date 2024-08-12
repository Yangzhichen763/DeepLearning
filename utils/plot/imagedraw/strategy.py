import math

import enum
from abc import ABC, abstractmethod
from typing import Callable


class DrawableObject:
    def __init__(self, scale):
        self.scale = scale

    def get_thickness(self, image, min_thickness=1):
        thickness = math.sqrt(image.shape[0] * image.shape[0] + image.shape[1] * image.shape[1])
        thickness = max(thickness, min_thickness)
        return int(self.scale * thickness)

    def get_trigger_range(self, image, trigger_range=10):
        return self.get_thickness(image, min_thickness=trigger_range) * 2


class EnsureState(enum.Enum):
    NONE = enum.auto()
    YES = enum.auto()
    NO = enum.auto()

    Next = enum.auto()
    DELETE = enum.auto()


class StrokeType(enum.Enum):
    NONE = enum.auto()
    POINT = enum.auto()
    LINE = enum.auto()


class ActionState(enum.Enum):
    NONE = enum.auto()
    SELECTING = enum.auto()
    SELECTED = enum.auto()
    IGNORED = enum.auto()
    DRAWING = enum.auto()


class Event:
    def __init__(self,
                 on_down: Callable[[str, object, dict], None] = None,
                 on_hold: Callable[[str, object, dict], None] = None,
                 on_drag: Callable[[str, object, dict], None] = None,
                 on_up: Callable[[str, object, dict], None] = None,
                 on_yes: Callable[[str, object, dict], None] = None,
                 on_no: Callable[[str, object, dict], None] = None,
                 on_ensure: Callable[[EnsureState, str, object, dict], None] = None,
                 on_default: Callable[[str, object, dict], None] = None):
        self.on_down = on_down
        self.on_hold = on_hold
        self.on_drag = on_drag
        self.on_up = on_up
        self.on_yes = on_yes
        self.on_no = on_no
        self.on_ensure = on_ensure
        self.on_default = on_default

    @staticmethod
    def on_up(on_up):
        return Event(on_up=on_up)

    @staticmethod
    def on_down(on_down):
        return Event(on_down=on_down)

    @staticmethod
    def on_hold(on_hold):
        return Event(on_hold=on_hold)

    @staticmethod
    def on_drag(on_drag):
        return Event(on_drag=on_drag)

    @staticmethod
    def empty():
        return Event()


class Strategy(ABC):
    # 线宽或绘制大小倍率，这个倍率随图像大小变化而变化
    line = DrawableObject(scale=1)
    point = DrawableObject(scale=4)
    text = DrawableObject(scale=0.5)
    arrow = DrawableObject(scale=2)

    # 事件
    on_ensure: Callable[[], EnsureState] = None
    on_image_initialize: Callable[[str, object], None] = None
    on_image_display: Callable[[str, object], None] = None
    on_image_close: Callable[[str], None] = None

    def __init__(self):
        pass

    @abstractmethod
    def process(self, path, image, event: Event, **kwargs):
        pass

    def ensure(self, path, image, event: Event, **kwargs):
        """
        等待按键按下，“确认” 或者 “取消” 操作
        """
        ensure_state = self.on_ensure() if self.on_ensure is not None else EnsureState.NONE

        if ensure_state == EnsureState.YES:
            event.on_yes(path, image, **kwargs)
        elif ensure_state == EnsureState.NO:
            event.on_no(path, image, **kwargs)

        if ensure_state != EnsureState.NONE:
            event.on_ensure(ensure_state, path, image, **kwargs)

        return ensure_state

# == 图像处理、绘制和展示 ==
    @staticmethod
    def image_initialize(window_name: str, image):
        """
        在图像刚传入时调用，用来初始化图像
        """
        if Strategy.on_image_initialize is not None:
            Strategy.on_image_initialize(window_name, image)

    @staticmethod
    def image_display(window_name: str, image):
        """
        在显示图像时调用，用来显示图像
        """
        if Strategy.on_image_display is not None:
            Strategy.on_image_display(window_name, image)

    @staticmethod
    def image_close(window_name: str):
        """
        关闭图像
        """
        if Strategy.on_image_close is not None:
            Strategy.on_image_close(window_name)

    @staticmethod
    def color_map(stroke_type: StrokeType, action_state: ActionState):
        """
        对应 StrokeType 和 ActionState 的颜色映射
        """
        if action_state == ActionState.DRAWING:
            return "#FFFFFF"
        elif action_state == ActionState.SELECTED:
            if stroke_type == StrokeType.POINT:
                return "#FFEB3C"
            elif stroke_type == StrokeType.LINE:
                return "#FCBF07"
        elif action_state == ActionState.SELECTING:
            if stroke_type == StrokeType.POINT:
                return "#FFEB3C"
            elif stroke_type == StrokeType.LINE:
                return "#FCBF07"
        elif action_state == ActionState.IGNORED:
            if stroke_type == StrokeType.POINT:
                return "#9D9648"
            elif stroke_type == StrokeType.LINE:
                return "#A3842E"
        else:
            return "#FF0000"

    @staticmethod
    def set_parameter(image, stroke_type: StrokeType, action_state: ActionState):
        """
        根据 StrokeType 和 ActionState 获取线宽和颜色
        """
        if stroke_type == StrokeType.POINT:
            thickness = Strategy.point.get_thickness(image)
        elif stroke_type == StrokeType.LINE:
            thickness = Strategy.line.get_thickness(image)
        else:
            thickness = DrawableObject(scale=1).get_thickness(image)

        color = Strategy.color_map(stroke_type, action_state)
        return thickness, color

    @staticmethod
    def get_trigger_point_index(image, points, mouse_point):
        """
        获取 mouse_point 在 points 中的最近并且满足触发条件的点的索引（仅包括点索引）
        """
        def get_distance(a, b):
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        trigger_index = 0
        min_distance = float('inf')
        trigger_range = Strategy.point.get_trigger_range(image)

        for i, point in enumerate(points):
            distance = get_distance(point, mouse_point)
            if distance < min_distance:
                min_distance = distance
                trigger_index = i

        if min_distance <= trigger_range:
            return trigger_index
        else:
            return None

    @staticmethod
    def get_trigger_point_group(image, point_groups, mouse_point):
        """
        获取 mouse_point 在 point_groups 中的最近并且满足触发条件的点的索引（包括组索引和点索引）
        """
        def get_distance(a, b):
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        trigger_index = dict(group=0, index=0)
        min_distance = float('inf')
        trigger_range = Strategy.point.get_trigger_range(image)

        for i_group, point_group in enumerate(point_groups):
            for i_index, point in enumerate(point_group):
                distance = get_distance(point, mouse_point)
                if distance < min_distance:
                    min_distance = distance
                    trigger_index = dict(group=i_group, index=i_index)

        if min_distance <= trigger_range:
            return trigger_index
        else:
            return None





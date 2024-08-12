import copy
from abc import abstractmethod, ABC

import sys
sys.path.append("..")
# noinspection PyUnresolvedReferences
from strategy import Strategy, Event
# noinspection PyUnresolvedReferences
from manager import Manager
sys.path.pop()


class MouseStrategy(Strategy, ABC):
    def __init__(self,
                 display_window_name: str,
                 output_window_name: str):
        super().__init__()
        self.path = None
        self.image = None
        self.event = None
        self.kwargs = None

        self.display_image = None
        self.output_image = None
        self.display_window_name = display_window_name
        self.output_window_name = output_window_name

    def process(self, path, image, event: Event, **kwargs):
        self.path = path
        self.image = image
        self.event = event
        self.kwargs = kwargs
        self.display_image = copy.deepcopy(image)
        self.output_image = copy.deepcopy(image)

        Manager.clear_mouse_callback()
        Manager.on_mouse_down = self.on_mouse_down
        Manager.on_mouse_up = self.on_mouse_up
        Manager.on_mouse_move = self.on_mouse_move
        Manager.on_mouse_wheel = self.on_mouse_wheel

        self.image_display(self.output_window_name, self.output_image)
        self.image_initialize(self.display_window_name, self.display_image)

    @abstractmethod
    def on_mouse_down(self):
        pass

    @abstractmethod
    def on_mouse_up(self):
        pass

    @abstractmethod
    def on_mouse_move(self):
        pass

    @abstractmethod
    def on_mouse_wheel(self):
        pass

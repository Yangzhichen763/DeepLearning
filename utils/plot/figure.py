import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import enum

# 所有 Color 类型，参考：https://matplotlib.org/stable/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py
# 所有 ColorMap 取值，参考：https://matplotlib.org/stable/users/explain/colors/colormaps.html
# ... #


# 所有 LineStyle 类型，参考：https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html#linestyles
class LineStyle(enum.Enum):
    EMPTY = " "      # 无线条
    SOLID = "-"      # 实线
    DOTTED = ":"     # 点线
    DASHED = "--"    # 虚线
    DASH_DOT = "-."  # 点划线


# 所有 Marker 类型，参考：https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
class Marker(enum.Enum):
    DOT = "."               # 小圆点
    PIXEL = ","             # 像素点
    CIRCLE = "o"            # 实心圆
    TRIANGLE_UP = "^"       # 正三角形
    TRIANGLE_DOWN = "v"     # 倒三角形
    TRIANGLE_LEFT = "<"     # 左三角形
    TRIANGLE_RIGHT = ">"    # 右三角形
    TRI_DOWN = "1"          # 向下三角形骨架
    TRI_UP = "2"            # 向上三角形骨架
    TRI_LEFT = "3"          # 向左三角形骨架
    TRI_RIGHT = "4"         # 向右三角形骨架
    OCTAGON = "8"           # 八边形
    SQUARE = "s"            # 正方形
    PENTAGON = "p"          # 五边形
    STAR = "*"              # 五角星
    HEXAGON_1 = "h"         # 尖角竖向六边形
    HEXAGON_2 = "H"         # 尖角横向六边形
    PLUS = "+"              # 加号
    PLUS_FILLED = "P"       # 加号（实心）
    X = "x"                 # 叉号
    X_FILLED = "X"          # 叉号（实心）
    DIAMOND_THIN = "d"      # 瘦菱形
    DIAMOND = "D"           # 方菱形
    VLINE = "|"             # 竖线
    HLINE = "_"             # 横线


# 所有 Hatch 类型，参考：https://matplotlib.org/stable/gallery/shapes_and_collections/hatch_style_reference.html
class Hatch(enum.Enum):
    SOLID = ""          # 无填充线
    LEFT_SLASH = "/"    # 左斜线
    RIGHT_SLASH = "\\"  # 右斜线
    CROSS = "x"         # 交叉十字字线
    PLUS = "+"          # 十字线
    VERTICAL = "|"      # 竖线
    HORIZONTAL = "-"    # 横线
    STAR = "*"          # 五角星
    POINT = "."         # 实心圆
    CIRCLE_SMALL = "o"  # 小空心圆
    CIRCLE_LARGE = "O"  # 大空心圆


class Location(enum.Enum):
    BEST = "best"
    UPPER_RIGHT = "upper right"
    UPPER_LEFT = "upper left"
    LOWER_LEFT = "lower left"
    LOWER_RIGHT = "lower right"
    RIGHT = "right"
    CENTER_LEFT = "center left"
    CENTER_RIGHT = "center right"
    LOWER_CENTER = "lower center"
    UPPER_CENTER = "upper center"
    CENTER = "center"


class Align(enum.Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


class FontWeight(enum.Enum):
    LIGHT = "light"
    NORMAL = "normal"
    MEDIUM = "medium"
    SEMIBOLD = "semibold"
    BOLD = "bold"
    HEAVY = "heavy"
    BLACK = "black"


class FontStyle(enum.Enum):
    NORMAL = "normal"
    ITALIC = "italic"
    OBLIQUE = "oblique"




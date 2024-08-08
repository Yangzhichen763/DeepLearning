import re

from matplotlib import pyplot as plot


def colored_text(ax, x, y, s, default_color="black", **kwargs):
    """
    Transform the color html code in the text to the color that can be displayed in the terminal.

    Args:
        ax (plot.Axes): The axes to add the text
        x (float): The x position of the text
        y (float): The y position of the text
        s (str): The text to be displayed
        default_color (str): The default color of the text
        **kwargs: Additional kwargs are passed to `Text` and `annotate` functions of matplotlib.

    Returns:
        The text axes
    """
    texts, colors = extract_text_and_colors(s, default_color)
    for i, (text, color) in enumerate(zip(texts, colors)):
        if i == 0:
            if "xy" in kwargs.keys():
                kwargs.pop("xy")
            ax = plot.text(x, y, text, color=color, **kwargs)
        else:
            if "xy" not in kwargs.keys():
                kwargs["xy"] = (1, 0)
            if "va" not in kwargs.keys():
                kwargs["va"] = "bottom"
            if "ha" not in kwargs.keys():
                kwargs["ha"] = "left"
            ax = plot.annotate(text=text, xycoords=ax,  color=color, **kwargs)

    return ax


def custom_colored_text(ax, x, y, s, default_color="black", color_kwarg_definitions=None):
    """
    Transform the color html code in the text to the color that can be displayed in the terminal.

    Args:
        ax (plot.Axes): The axes to add the text
        x (float): The x position of the text
        y (float): The y position of the text
        s (str): The text to be displayed
        default_color (str): The default color of the text
        color_kwarg_definitions: A dictionary of color definitions. The keys are the color names,
            and the values are the kwargs for the `Text` and `annotate` functions of matplotlib.

    Returns:
        The text axes
    """
    texts, colors = extract_text_and_colors(s, default_color)
    for i, (text, color) in enumerate(zip(texts, colors)):
        if color in color_kwarg_definitions.keys():
            kwargs = color_kwarg_definitions[color]
        else:
            kwargs = {}

        if i == 0:
            if "xy" in kwargs.keys():
                kwargs.pop("xy")
            ax = plot.text(x, y, text, color=color, **kwargs)
        else:
            if "xy" not in kwargs.keys():
                kwargs["xy"] = (1, 0)
            if "va" not in kwargs.keys():
                kwargs["va"] = "bottom"
            if "ha" not in kwargs.keys():
                kwargs["ha"] = "left"
            ax = plot.annotate(text=text, xycoords=ax, color=color, **kwargs)

    return ax

def extract_text_and_colors(text, default_color=None) -> (list[str], list[str]):
    """
    Extract text and colors from a string.
    The string can contain html tags to specify the color of the text.
    e.g. <c=red>red</c>, <red>text</red>, <red>text</>
    e.g. <c=#FF0000>red</c>, <#FF0000>text</#FF0000>, <#FF0000>text</>

    Args:
        text (str): The input string.
            e.g.: "This is <red>red</red> text." -> ["This is ", "red", " text."], [default_color, "red", default_color]
        default_color (str): The default color of the text.

    Returns:
        A tuple of two lists: the first list contains the text of each segment, and the second list contains the color of each segment.
    """
    tag_patterns = [r'<([^/].*?)>(.*?)</\1>', r'<c=([^/].*?)>(.*?)</c>', r'<([^/].*?)>(.*?)</>']
    text_pattern = r'([^<>]*)(<[^/>].*?>.*?</.*?>)?'

    texts = re.findall(text_pattern, text)

    split_texts = []
    colors = []
    for outer, inner in texts:
        # 排除空字符串的情况
        if outer == '' and inner == '':
            continue

        # 处理没有颜色标签的字符串
        colors.append(default_color)
        split_texts.append(outer)

        # 处理带有颜色标签的字符串
        for tag_pattern in tag_patterns:
            result = re.search(tag_pattern, inner)
            if result is not None:
                colors.append(result.group(1))
                split_texts.append(result.group(2))
                break

    return split_texts, colors


def format_text(ax, x, y, s, default_textprops: dict = None, highlight_textprops: list[dict] = None):
    """
    Transform the format html code in the text to the format that can be displayed in the terminal.

    Args:
        ax (plot.Axes): The axes to add the text
        x (float): The x position of the text
        y (float): The y position of the text
        s (str): The text to be displayed
        default_textprops (dict): The default text properties of the text.
        highlight_textprops (list[dict]): The text properties of the text to be highlighted.

    Returns:
        The text axes
    """
    texts, formats = extract_formatted_text(s)
    i_format = 0
    for i, (text, any_format) in enumerate(zip(texts, formats)):
        textprops = default_textprops if any_format is False else highlight_textprops[i_format]
        textprops = textprops if textprops is not None else {}
        if i == 0:
            if "xy" in textprops.keys():
                textprops.pop("xy")
            ax = plot.text(x, y, text, **textprops)
        else:
            if "xy" not in textprops.keys():
                textprops["xy"] = (1, 0)
            if "va" not in textprops.keys():
                textprops["va"] = "bottom"
            if "ha" not in textprops.keys():
                textprops["ha"] = "left"
            ax = plot.annotate(text=text, xycoords=ax, **textprops)
        if any_format is True:
            i_format += 1

    return ax

def extract_formatted_text(text) -> (list[str], list[bool]):
    """
    Extract formatted text from a string.
    The string can contain "<", ">" tags to specify the format of the text.
    e.g. <text> -> format: True, text -> format: False

    Args:
        text (str): The input string.
            e.g.: "This is <red> text." -> ["This is ", "red", " text."], [False, True, False]

    Returns:
        A tuple of two lists: the first list contains the text of each segment,
        and the second list contains the format of each segment.
    """
    tag_patterns = [r'<(.*?)>']
    text_pattern = r'([^<>]*)(<[^/>].*?>.*?)?'

    texts = re.findall(text_pattern, text)

    split_texts = []
    formats = []
    for outer, inner in texts:
        # 排除空字符串的情况
        if outer == '' and inner == '':
            continue

        # 处理没有颜色标签的字符串
        formats.append(False)
        split_texts.append(outer)

        # 处理带有颜色标签的字符串
        for tag_pattern in tag_patterns:
            result = re.search(tag_pattern, inner)
            if result is not None:
                formats.append(True)
                split_texts.append(result.group(1))
                break

    return split_texts, formats


if __name__ == '__main__':
    _text = r'This is a text with <c=red>red</c>, <blue>blue</blue> and <green>green</> colors.'
    _split_texts, _colors = extract_text_and_colors(_text)
    print(_split_texts, _colors)

    _text = r'This is a text with <red>, <blue> and <green> colors.'
    _split_texts, _formats = extract_formatted_text(_text)
    print(_split_texts, _formats)

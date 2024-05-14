import os


def get_root_path():
    """
    寻找项目的根目录路径
    """
    path = __file__
    min_relative_path_length = len(os.path.relpath(path))
    while True:
        next_path = os.path.dirname(path)
        current_relative_path_length = len(os.path.relpath(next_path))
        if current_relative_path_length >= min_relative_path_length:
            break

        path = next_path
        min_relative_path_length = min(current_relative_path_length, min_relative_path_length)

    return path

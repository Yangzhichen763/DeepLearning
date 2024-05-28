import os


def get_root_path():
    """
    寻找项目的根目录路径
    """
    path = __file__
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.dirname(path)

    return path


def get_unique_file_name(directory, file_name, suffix, unique=True):
    """
    获取一个唯一的文件名
    Args:
        directory: 文件夹路径
        file_name: 文件名
        suffix: 不带点的后缀名，比如jpg, png, txt等
        unique: 是否需要唯一化，如果为 False，则直接返回原文件名

    Returns:

    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not unique:
        return os.path.join(directory, f"{file_name}.{suffix}")

    k = 0
    while True:
        save_file_name = f"{file_name}_{k}.{suffix}"
        image_path = os.path.join(directory, save_file_name)
        if not os.path.exists(image_path):
            break
        k += 1

    return image_path


def get_unique_full_path(full_path, unique=True):
    """
    获取一个唯一的文件名
    Args:
        full_path: 文件路径
        unique: 是否需要唯一化，如果为 False，则直接返回原文件名

    Returns:

    """
    if not os.path.exists(os.path.dirname(full_path)):
        os.makedirs(os.path.dirname(full_path))

    if not unique:
        return full_path

    directory, file_name = os.path.split(full_path)
    file_name, suffix = os.path.splitext(file_name)
    k = 0
    while True:
        save_file_name = f"{file_name}_{k}.{suffix}"
        image_path = os.path.join(directory, save_file_name)
        if not os.path.exists(image_path):
            break
        k += 1

    return image_path

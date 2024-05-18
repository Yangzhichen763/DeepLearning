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


def get_unique_file_name(dir, file_name, suffix, unique=True):
    """
    获取一个唯一的文件名
    Args:
        dir: 文件夹路径
        file_name: 文件名
        suffix: 不带点的后缀名，比如jpg, png, txt等

    Returns:

    """
    if not os.path.exists(dir):
        os.makedirs(dir)

    if not unique:
        return os.path.join(dir, f"{file_name}.{suffix}")

    k = 0
    while True:
        save_file_name = f"{file_name}_{k}.{suffix}"
        image_path = os.path.join(dir, save_file_name)
        if not os.path.exists(image_path):
            break
        k += 1

    return image_path

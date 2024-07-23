from tqdm import tqdm


def warning(message: str):
    print_(f"\033[1;31mWARNING⚠️: {message}\033[0m")


def error(message: str):
    print_(f"\033[1;31mERROR⚠️: {message}\033[0m")


def print_(*values: object, sep: str = ' ', end: str = '\n', **kwargs):
    output = sep.join(str(v) for v in values)
    tqdm.write(output, end=end, **kwargs)
import torch
from torch import nn
import numpy as np
from math import ceil, floor
from torchsummary import summary

from utils import logger
from utils.pytorch.deploy import assert_on_cuda

indent = 0


# ------------------------------ log_sequential_model_info() ---------------------------------

def log_sequential_model_info(model, x_input):
    """
    输出模型信息
    \n如果模型中含有 nn.Sequential，则遍历其每一层，并输出其名字和输出数据的形状
    :param model: 要输出信息的模型
    :param x_input: 输入数据，用于获取模型的输入和输出数据形状
    :return:
    """
    global indent
    indent = 0
    log_model_header(model)
    # 遍历输出模型的每一层，并输出其名字和输出数据的形状
    for name, layer in model.named_children():
        # 如果是 nn.Sequential 则遍历其每一层，否则直接输出其输出数据的形状
        if isinstance(layer, nn.Sequential):
            log_layer_params(name, layer, x_input, any_inner_layer=True)
            for name_sub, layer_sub in layer.named_children():
                x_input = layer_sub(x_input)
                log_layer_params(name_sub, layer_sub, x_input)
            log_end_inner_layer()
        else:
            x_input = layer(x_input)
            log_layer_params(name, layer, x_input)
    log_end_inner_layer()


def log_model_header(model):
    global indent
    print("%s{" % model.__class__.__name__)
    indent += 1


def log_layer_params(name, layer, x_input, any_inner_layer=False):
    """
    输出层的名字或序号、层的类名和输出数据的形状
    Args:
        name (str): 层的名字或序号
        layer: 要输出信息的层
        x_input: 输入数据，用于获取层的输出数据形状
        any_inner_layer: 如果设置该值为 True，则输出层的名字时会加上括号；并且需要手动在输出结束时调用 log_end_inner_layer()
    """
    head = "%s (%s):" % (get_indent_blanks(), name)
    if any_inner_layer:
        tail = "%s{" % layer.__class__.__name__
        global indent
        indent += 1
    else:
        class_name = layer.__class__.__name__
        shape_log = get_shape_log(x_input.shape)
        padding = '\t' * ceil((43 - head.__len__() - class_name.__len__()) / 4)
        tail = "%s%s< %s" % (class_name, padding, shape_log)
    print(head, tail)


def log_end_inner_layer():
    global indent
    indent -= 1
    print(get_indent_blanks(), '}')


def get_indent_blanks():
    return '  ' * indent


def get_shape_log(shape):
    return "[%s]" % ', '.join(str(i) for i in shape)


# ------------------------------ log_model_params() ---------------------------------

def log_model_params(model, input_shape):
    """
    输出模型参数信息
    Args:
        model: 要输出信息的模型
        input_shape (torch.Size): 输入形状，比如 (1, 3, 224, 224)
    """
    if input_shape is None:
        logger.warning("input cannot be None.")
        exit()
    elif not isinstance(input_shape, torch.Size):
        logger.warning("input cannot be not a tensor or a tensor shape.")
        exit()

    device = assert_on_cuda()
    model = model.to(device)
    summary(model=model,
            input_size=tuple(input_shape)[1:],  # tuple(input_shape) 是 (1, 3, 224, 224)
            # batch_size=input_shape[0],
            device=device.__str__())

    x_input = torch.Tensor(*input_shape)        # *input_shape 是 1 3 224 224
    x_input = x_input.to(device)
    print("Input shape: ", x_input.shape)

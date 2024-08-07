import torch
from torch import nn
from math import ceil, floor

# from torchsummary import summary    # 旧的 summary 加入 LSTM 之类的模型会报错，需要用新的 summary
from torchinfo import summary

from utils.log.info import print_
from utils.torch.deploy import assert_on_cuda

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
    print_("%s{" % model.__class__.__name__)
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
    print_(head, tail)


def log_end_inner_layer():
    global indent
    indent -= 1
    print_(get_indent_blanks(), '}')


def get_indent_blanks():
    return '  ' * indent


def get_shape_log(shape):
    return "[%s]" % ', '.join(str(i) for i in shape)


# ------------------------------ log_model_params() ---------------------------------

def log_model_params(model, **kwargs):
    """
    输出模型参数信息，kwargs 可选参数如下：
    \ninput_size (torch.Size | tuple): 输入形状，比如 (1, 3, 224, 224)
    \ninput_data (torch.tensor): 输入张量
    \n以及其他参数，参考 torchinfo.summary()
    Args:
        model: 要输出信息的模型
    """
    if kwargs.get('input_size') is not None:
        input_size: tuple[int] = kwargs.get('input_size')
    elif kwargs.get('input_data') is not None:
        input_data: torch.Tensor = kwargs.get('input_data')
        input_size = input_data.shape
    else:
        raise ValueError("input_size or input_data should be provided.")

    device = assert_on_cuda()
    model = model.to(device)
    summary(model=model,
            batch_dim=0,
            device=device.__str__(),
            **kwargs)

    x_input = torch.Tensor(*input_size)        # *input_shape 是 1 3 224 224
    x_input = x_input.to(device)
    print_("Input shape: ", x_input.shape)
    y_output = model(x_input)
    if isinstance(y_output, torch.Tensor):
        print_("Output shape: ", y_output.shape)
    elif isinstance(y_output, (list, tuple)):
        for i, y in enumerate(y_output):
            print_(f"Output {i} shape: ", y.shape)
    print_('\n')

import torch


def test_model(model, x_input, device):
    """
    测试模型.
    :param model:
    :param x_input:
    :param device:
    :return:
    """
    model.eval()
    with torch.no_grad():
        x_input = x_input.to(device)
        output = model(x_input)
        return output.argmax(dim=1, keepdim=True)

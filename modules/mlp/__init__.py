import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    参考代码：https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ):
        """
        Args:
            input_dim (int): input dimension
            hidden_dim (int): hidden dimension
            output_dim (int): output dimension
            num_layers (int): number of layers
            activation (nn.Module, optional): activation function. Defaults to nn.ReLU.
            sigmoid_output (bool, optional): whether to apply sigmoid activation to output. Defaults to False.
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

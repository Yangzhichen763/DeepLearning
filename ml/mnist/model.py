
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes: [list, int]=128, num_classes=10):
        super(MLP, self).__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        in_feature = input_size
        out_feature = hidden_sizes[0]

        modules = []
        for i in range(len(hidden_sizes)):
            modules.append(nn.Linear(in_feature, out_feature))

            if i < len(hidden_sizes) - 1:
                in_feature = out_feature
                out_feature = hidden_sizes[i + 1]

        self.layers = nn.Sequential(*modules)
        self.output_layer = nn.Linear(in_feature, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = nn.functional.relu(layer(x))

        return x


if __name__ == '__main__':
    model = MLP(input_size=784, hidden_sizes=[128, 64], num_classes=10)
    print(model)

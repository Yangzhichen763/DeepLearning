import torch
from torch.autograd import Variable
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.linear(D_in, H)
        self.linear2 = nn.linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = TwoLayerNet(D_in, H, D_out)

criterion = nn.MSELoss(size_average=False)                  # 均方误差损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

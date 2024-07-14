import torch
import torch.nn as nn
import torch.nn.functional as F


class MOELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, top_k=2):
        super(MOELayer, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        gate_score = F.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.bmm(gate_score.unsqueeze(1), expert_outputs).squeeze(1)
        return output


if __name__ == '__main__':
    input_dim = 5
    output_dim = 3
    num_experts = 4
    batch_size = 2

    moe_layer = MOELayer(input_dim, output_dim, num_experts)
    x = torch.randn(batch_size, 5)
    y = moe_layer(x)
    print(y.shape)

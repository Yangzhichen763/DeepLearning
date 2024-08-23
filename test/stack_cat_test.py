import torch

# stack((a, b), dim=-1).flatten(start_dim=-2) <==shape=equal==> cat((a, b), dim=-1)

a = torch.tensor([[[[1, 2, 3], [4, 5, 6]]]])
b = torch.tensor([[[[-1, -2, -3], [-4, -5, -6]]]])

dim_a = -1
dim_b = dim_a - 1

c = torch.stack((a, b), dim=dim_a)
print(c)

_c = c.flatten(start_dim=dim_b)
print(_c)

d = torch.cat((a, b), dim=dim_a)
print(d)

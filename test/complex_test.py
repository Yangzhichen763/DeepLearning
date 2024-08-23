import torch


a = torch.tensor([1, 2, 3, 4]).float()
b = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).float()
c = torch.complex(real=a, imag=torch.zeros_like(a))
d = torch.view_as_complex(b.float().reshape(*b.shape[:-1], -1, 2))
x = torch.view_as_real(c * d).flatten(start_dim=-2)

print(a)
print(b)
print(c)
print(d)
print(x)

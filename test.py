import torch

a = 3.6
b = -2.312
c = 2

a = torch.tensor(a).double()
a.requires_grad = True
b = torch.tensor(b).double()
b.requires_grad = True
c = torch.tensor(c).double()
c.requires_grad = True

l = torch.abs((a-b)/c) - a

l.backward()

print(f"loss: {l}")
print(f"a.grad: {a.grad}, b.grad: {b.grad}, c.grad: {c.grad}")


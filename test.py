import torch
from Atom import Atom

x = Atom(-4.0)
z = 2 * x + 2 + x
q = z.relu() + z * x
h = (z * z).relu()
y = h + q + q * x
y.backward()
xmg, ymg = x, y

x = torch.tensor(-4.0, requires_grad=True)
x.requires_grad = True
z = 2 * x + 2 + x
q = z.relu() + z * x
h = (z * z).relu()
y = h + q + q * x
y.backward()
xpt, ypt = x, y

assert(xmg.grad == xpt.grad)
assert(ymg.data == ypt.data)
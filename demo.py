from Atom import Atom
from nn import MLP, Layer, Neuron

x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])

# Create a dataset
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]

# Train
for k in range(1000):
    # forward
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) * (yout - ygt) for yout, ygt in zip(ys, ypred))

    # Backward 
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # Update
    for p in n.parameters():
        p.data -= 0.005 * p.grad

    print(k, loss.data)
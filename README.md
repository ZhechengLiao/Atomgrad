# Atomgrad
<p align="center">
<img src="/atomgrad.png" width=200 >
</p>

## Introduction
Atomgrad is a simple version AI frame work like pytorch following [micrograd](https://github.com/karpathy/micrograd). Build from scratch to implement neuron network training and a lot of deep learning alogrithm, including transformers, GPT, stable diffusion, etc...(This is only for learning deep learning)

## ToDo
- [x] Basic data structure for neuron network
- [x] Data structure for neuron network
- [ ] Transformers
- [ ] Stable Diffusion
- [ ] GPT model
- [ ] ...

## Example
### Basic Engine
```python
# Forward propagation
x = Atom(-4.0)
z = 2 * x + 2 + x
q = z.relu() + z * x
h = (z * z).relu()
y = h + q + q * x
print(y) # Atom(data=-20.0, grad=1.0)

# Backward propagation
y.backward() # calculate all gradient for input
```
### Tiny Demo
```python
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
```



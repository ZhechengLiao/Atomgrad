# Atomgrad
<p align="center">
<img src="/atomgrad.png" width=200 >
</p>

## Introduction
Atomgrad is a simple version AI frame work like pytorch. Build from scratch to implement neuron network training and a lot of deep learning alogrithm, including transformers, GPT, stable diffusion, etc...(This is only for learning deep learning)

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



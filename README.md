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
a = Atom(4.0)
b = Atom(3.0)
c = a + b 
print(c) # Atom(data=7.0, grad=0.0)
```



class Atom:
    def __init__(self, data, _children=(), _opt=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None # initially do nothing
        self._prev = set(_children)
        self.opt = _opt
        
    def __add__(self, other):
        other = other if isinstance(other, Atom) else Atom(other)
        out = Atom(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Atom) else Atom(other)
        out = Atom(self.data * other.data, (self, other), '*')
        def _backward():
            # calculus: chain rule
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad 
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support int/float power"
        out = Atom(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out
    
    def relu(self):
        out = Atom(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    
    def __repr__(self):
        return f"Atom(data={self.data}, grad={self.grad})"
    
    def __neg__(self):
        return self * -1
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return self + (-other)
    
    def __radd__(self, other):
        return self + other
    
    

    

   
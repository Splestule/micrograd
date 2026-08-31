import math

class Value:
    def __init__(self, data, _children=()):
        self.data = data
        self._prev = set(_children)
        self._backward = lambda: None
        self.grad = 0.0
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data, (self, other))
        
        def _backward():
            other.grad += self.data * out.grad
            self.grad += other.data * out.grad
        out._backward = _backward
        
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1
    
    def __rtruediv__(self, other):
        return self**-1 * other
    
    def __rmul__(self, other):
        return self * other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __neg__(self):
        return -1*self

    def __pow__(self, x):
        out = Value(self.data**x, (self,))
        
        def _backward():
            self.grad += x * self.data**(x-1) * out.grad
        out._backward = _backward
        
        return out

    def tanh(self):
        e = math.exp(2*self.data)
        t = (e-1)/(e+1)
        out = Value(t, (self,))
        
        def _backward():
            self.grad +=  (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def backward(self):
        topo = list()
        visited = set()
        def topo_build(node):
            assert isinstance(node, Value)
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    topo_build(child)
                topo.append(node)
        topo_build(self,)
        queue = reversed(topo)
        self.grad = 1.0
        for node in queue:
            assert isinstance(node, Value)
            node._backward()
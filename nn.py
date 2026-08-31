import numpy as np
from value import Value

class Neuron:
    def __init__(self, nin):
        self.nin = nin
        self.w = [Value(np.random.uniform(-1,1)) for _ in range(self.nin)]
        self.b = Value(np.random.uniform(-1,1))
        
    def parameters(self): return self.w + [self.b]

    def __call__(self, xs=None):
        assert len(xs) == self.nin
        return sum([wi*xi for wi, xi in zip(self.w, xs)], self.b).tanh()
    
class Layer:
    def __init__(self, nin, nouts):
        self.nin = nin
        self.nouts = nouts
        self.neurons = [Neuron(self.nin) for _ in range(nouts)]
        
    def parameters(self): return [p for n in self.neurons for p in n.parameters()]
        
    def __call__(self, xs=None):
        assert len(xs) == self.nin
        return [n(xs) for n in self.neurons]
    
class MLP:
    def __init__(self, nin, nouts):
        assert isinstance(nouts, list)
        self.nin = nin
        self.nouts = nouts
        self.structure = [self.nin] + nouts
        self.layers = [Layer(a, b) for a, b in zip(self.structure, self.structure[1:])]
        
    def parameters(self): return [p for l in self.layers for p in l.parameters()]
            
    def __call__(self, xs=None):
        assert len(xs) == self.nin
        activations = xs.copy()
        for layer in self.layers:
            activations = layer(activations)
        return activations
    
    def loss(self, xs=None, ys=None):
        last_layer_out = self(xs)
        return sum((y - b)**2 for b, y in zip(last_layer_out, ys))

    def zerograd(self):
        for parameter in self.parameters():
            parameter.grad = 0.0
                
    def train(self, iterations, learning_rate=0.1, xs=None, ys=None):
        for _ in range(iterations):
            self.zerograd()
            loss = sum(self.loss(x, y) for x, y in zip(xs, ys))/(len(xs))
            loss.backward()
            for parameter in self.parameters():
                parameter.data -= learning_rate*parameter.grad
        return loss
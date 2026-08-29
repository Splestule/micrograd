import math
import numpy as np
from value import Value

class Neuron:
    def __init__(self, nin):
        self.nin = nin
        self.w = [Value(np.random(-1,1)) for _ in range(self.nin)]
        self.b = Value(np.random(-1,1))
        self.parameters = self.w.extend([self.b])
        
    def __call__(self, xs=[]):
        assert len(xs) == self.nin
        return sum([wi*xi for wi, xi in zip(self.w, xs)], self.b).tanh
    
class Layer:
    def __init__(self, nin, nouts):
        self.nin = nin
        self.nouts = nouts
        self.neurons = [Neuron(self.nin) for _ in range(nouts)]
        
    def __call__(self, xs=[]):
        assert len(xs) == self.nin
        return [n(x) for n, x in zip(self.neurons, xs)]
    
class MLP:
    def __init__(self, nin, nouts):
        assert isinstance(nouts, list)
        self.nin = nin
        self.nouts = nouts
        self.structure = [self.nin].extend(nouts)
        self.layers = [Layer(self.structure[i], self.structure[i+1]) for i in range(self.nouts)]
        
    #forward pass
    
    #backward pass
    
    #loss output/function
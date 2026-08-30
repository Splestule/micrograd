import math
import numpy as np
from value import Value

LEARNING_RATE = .5

class Neuron:
    def __init__(self, nin):
        rng = np.random
        self.nin = nin
        self.w = [Value(rng.uniform(-1,1)) for _ in range(self.nin)]
        self.b = Value(rng.uniform(-1,1))
        self.parameters = self.w.copy()
        self.parameters.extend([self.b])
    def __call__(self, xs=None):
        assert len(xs) == self.nin
        return sum([wi*xi for wi, xi in zip(self.w, xs)], self.b).tanh()
    
class Layer:
    def __init__(self, nin, nouts):
        self.nin = nin
        self.nouts = nouts
        self.neurons = [Neuron(self.nin) for _ in range(nouts)]
        
    def __call__(self, xs=None):
        assert len(xs) == self.nin
        return [n(xs) for n in self.neurons]
    
class MLP:
    def __init__(self, nin, nouts):
        assert isinstance(nouts, list)
        self.nin = nin
        self.nouts = nouts
        self.structure = [self.nin]
        self.structure.extend(nouts)
        self.layers = [Layer(self.structure[i], self.structure[i+1]) for i in range(len(self.nouts))]
        
    def __call__(self, xs=None, ys=None):
        assert isinstance(xs, list) and isinstance(ys, list) and len(xs) == self.nin
        input = xs.copy()
        for layer in self.layers:
            input = layer(input)
        loss = sum((y - b)**2 for b, y in zip(input, ys))
        return loss, input

    def zerograd(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                for parameter in neuron.parameters:
                    parameter.grad = 0.0
                
    def train(self, iterations, xs=None, ys=None):
        for _ in range(iterations):
            self.zerograd()
            loss = sum(self(x, y)[0] for x, y in zip(xs, ys))/(len(xs))
            loss.backward()
            for layer in self.layers:
                for neuron in layer.neurons:
                    for parameter in neuron.parameters:
                        parameter.data -= LEARNING_RATE*parameter.grad
        return loss
                        
                        
xs = [[.5, .2, .6, .9, .5], [.5, .9, .4, .2, .1], [.9, .8, .11, .3, .4]]
ys = [[.1, -.5, .8, .4, .5], [.9, -.8, .4, .7, .6], [-.5, -.6, .4, .4, -.5]]
    
nn = MLP(5, [7, 8, 7, 5])
print(nn.train(2000, xs, ys))

for x in xs:
    _, output = nn(x, [])
    print(x, output)




# micrograd
A scalar-valued autograd engine and a small neural network library built on top of it.

Based on [Andrej Karpathy's video](https://www.youtube.com/watch?v=VMj-3S1tku0). I watched it once in a single sitting, then implemented everything from memory without rewatching.
This pet project made me learn the fundamental concepts behind the autograd architecture and experience some caveats such as saturating the tanh activation function by initializing the weights to be all-positive.

The repo contains the Value, Neuron, Layer and MLP classes enabling me to train very very simple neural networks. Of course this architecture isnt really built for production, but mainly just as a way for me to learn.

To try it out, run: ```python moons.py```

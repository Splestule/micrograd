from sklearn.datasets import make_moons
from nn import Neuron, Layer, MLP
import numpy as np
import matplotlib.pyplot as plt

dataset = make_moons(n_samples=200, noise=0.2, random_state=0)
xs = dataset[0].tolist()
ys = [[1.0] if v==1 else [-1.0] for v in dataset[1]]

nn = MLP(2, [5, 8, 1])
print(nn.train(300, xs=xs[:150], ys=ys[:150]))
loss = sum(nn.loss(x, y) for x, y in zip(xs[150:], ys[150:]))/(len(xs[150:]))
print(loss)

xx, yy = np.meshgrid(np.linspace(dataset[0][:,0].min(), dataset[0][:,0].max(), 50),
                     np.linspace(dataset[0][:,1].min(), dataset[0][:,1].max(), 50))

grid = np.c_[xx.ravel(), yy.ravel()]
Z = np.array([nn(pt)[0].data for pt in grid]).reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=['#c6dbef','#fcbba1'])
plt.contour(xx, yy, Z, levels=[0], colors='k')
plt.scatter(dataset[0][:,0], dataset[0][:,1], c=dataset[1], cmap='coolwarm', edgecolors='k')
plt.show()
import numpy as np
from nnfs.datasets import spiral_data
import random
import matplotlib.pyplot as plt
import nnfs
# to show the plot of the Data set
#
# X, y = spiral_data(samples= 100, classes=5)
# plt.scatter(X[:,0], X[:,1], c=y, cmap='brg') 
# plt.show()

# to calculate the layers of the neural network
#
# inputs = [[1.0, 2.0, 3.0, 2.5],[2.0, 5.0,-1.0, 2.0],[-1.5,2.7, 3.3,-0.8]]
# weights = [[0.2, 0.8, -0.5, 1.0],[0.5, -0.91, 0.26, -0.5],[-0.26, -0.27, 0.17, 0.87]]
# biases = [2.0, 3.0, 0.5]
# weights2 = [[0.1, -0.14, 0.5],[-0.5, 0.12, -0.331],[-0.44, 0.73, -0.131]]
# biases2 = [-1, 2, -0.5]
# layer1_outputs = np.dot (inputs, np.array (weights) .T) + biases
# layer2_outputs = np.dot (layer1_outputs, np.array (weights2) .T) + biases2

# print(layer2_outputs)

class Layer Dense:
    def _init__(self, n_inputs, n_neurons) :
        def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1, n_neurons))
    # Forward pass
    def forward(self, inputs):
    # Calculate output values from inputs, weights and biases
    pass # using pass statement as a placeholder
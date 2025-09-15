import numpy as np
from nnfs.datasets import spiral_data
import random
import matplotlib.pyplot as plt
import nnfs


# class Layer_Dense:
#     # Layer initialization
#     def __init__(self, n_inputs, n_neurons):
#         self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) 
#         self.biases = np.zeros((1, n_neurons))
#     # Forward pass
#     def forward(self, inputs):
#         self.output = np.dot(inputs, self.weights) + self.biases

# # Create dataset
# X, y = spiral_data(samples=100, classes=3)
# # Create Dense layer with 2 input features and 3 output values
# dense1 = Layer_Dense (2, 3)
# # Perform a forward pass of our training data through this layer
# dense1.forward(X)
# # Let's see output of the first few samples:
# print(dense1.output[:5])

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []
for i in inputs:
    output.append(max(0, i))
print(output)

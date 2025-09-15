import numpy as np
from nnfs.datasets import spiral_data
import random
import matplotlib.pyplot as plt
import nnfs

# ReLU activation
class Activation_ReLU:
    # Forward pass
    def forward (self, inputs):
        # Calculate output values from input
        self.output = np.maximum (0, inputs)

class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1, n_neurons))
    # Forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# # Create dataset
# X, y = spiral_data(samples=100, classes=3)
# # Create Dense layer with 2 input features and 3 output values
# dense1 = Layer_Dense (2, 3)
# # Create ReLU activation (to be used with Dense layer):
# activation1 = Activation_ReLU()
# # Make a forward pass of our training data through this layer
# dense1.forward(X)
# # Forward pass through activation func.
# # Takes in output from previous layer
# activation1.forward(dense1.output)
#  # Let's see output of the first few samples:
# print(activation1.output[:5])


layer_outputs = [4.8, 1.21, 2.385]
# e - mathematical constant, we use E here to match a common coding
# style where constants are uppercased
E = 2.71828182846 # you can also use math.e
# For each value in a vector, calculate the exponential value
exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output) # ** - power operator in Python
print("exponentiated values:")
print (exp_values)

# Now normalize values
norm_base = sum(exp_values) # We sum all values
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)
print('Normalized exponentiated values:')
print (norm_values)
print('Sum of normalized values:' , sum (norm_values))

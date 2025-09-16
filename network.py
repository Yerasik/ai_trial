import numpy as np
from nnfs.datasets import spiral_data
import nnfs


nnfs.init()

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

# Softmax activation
class Activation_Softmax:
# Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
# Common loss class
class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate (self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np. mean (sample_losses)
        # Return loss
        return data_loss

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len (y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
            range(samples), 
            y_true
        ]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
            y_pred_clipped * y_true,
            axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

# # Create dataset
# X, y = spiral_data(samples=100, classes=3)
# # Create Layers and Activations
# dense1 = Layer_Dense (2, 3)
# activation1 = Activation_ReLU()
# dense2 = Layer_Dense(3, 3)
# activation2 = Activation_Softmax()
# loss_function = Loss_CategoricalCrossentropy()

# # Pass the infroamtion and Normalize
# dense1.forward(X)
# activation1.forward(dense1.output)
# dense2.forward(activation1.output)
# activation2.forward(dense2.output)

# #print the result of the loss
# loss = loss_function.calculate (activation2.output, y)
# print(activation2.output[:5])
# print (loss)

# Probabilities of 3 samples
softmax_outputs = np.array([[0.7, 0.2, 0.1],
[0.5, 0.1, 0.4],
[0.02, 0.9, 0.08]])
# Target (ground-truth) labels for 3 samples
class_targets = np.array ([0, 1, 1])
# Calculate values along second axis (axis of index 1)
predictions = np.argmax (softmax_outputs, axis=1)
# If targets are one-hot encoded - convert them
if len(class_targets.shape) == 2:
    class_targets = np.argmax (class_targets, axis=1)
# True evaluates to 1; False to 0
accuracy = np.mean (predictions==class_targets)
print ('accuracy:', accuracy)


import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data

nnfs.init()
X, y = sine_data()
plt.plot(X, y)
plt.show()

# Linear activation
class Activation_Linear:
    # Forward pass
    def forward (self, inputs):
        # Just remember values
        self.inputs = inputs
        self.output = inputs
    # Backward pass
    def backward (self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()

# Mean Squared Error loss
class Loss_MeanSquaredError (Loss): # L2 loss
    # Forward pass
    def forward (self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean((y_true -y_pred)**2, axis=-1)
        # Return losses
        return sample_losses
    # Backward pass
    def backward (self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
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
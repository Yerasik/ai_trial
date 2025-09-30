import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data
import numpy as np 


# Dense layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons,weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros( (1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    # Forward pass
    def forward (self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
    # Backward pass
    def backward (self, dvalues) :
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0 :
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0 ] = - 1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0 :
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        
        # L1 on biases
        if self.bias_regularizer_l1 > 0 :
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0 ] = - 1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0 :
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

# ReLU activation
class Activation_ReLU:
    # Forward pass
    def forward (self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum (0, inputs)
    # Backward pass
    def backward (self, dvalues) :
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


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

# Common loss class
class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate (self, output, y) :
        # Calculate sample losses
        sample_losses = self.forward (output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss
    # Regularization loss calculation
    def regularization_loss ( self , layer ):
        # 0 by default
        regularization_loss = 0

        # Calculate only if the layer parameter is greater than o
        # L1 regulization
        if layer.weight_regularizer_l1>0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

        # L2 regulization
        if layer.weight_regularizer_l2>0:
            regularization_loss += layer.weight_regularizer_l2*np.sum(layer.weights*layer.weights)
        
        # L1 regulization biases
        if layer.bias_regularizer_l1>0:
            regularization_loss += layer.bias_regularizer_l1*np.sum(np.abs(layer.biases))

        # L2 regulization biases
        if layer.bias_regularizer_l2>0:
            regularization_loss += layer.bias_regularizer_l2*np.sum(layer.biases*layer.biases)
        
        return regularization_loss

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

# Mean Absolute Error loss
class Loss_MeanAbsoluteError (Loss): # L1 loss
    def forward (self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean(np.abs(y_true -y_pred), axis=-1)
        # Return losses
        return sample_losses
    # Backward pass
    def backward (self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues [0])
        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Optimizer_Adam:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params (self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self. decay * self.iterations))
    # Update parameters
    def update_params (self, layer):
        # If there is no cache array for weights
        # create them
        if not hasattr (layer,'weight_cache' ):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        # Update momentums with squared current gradients
        layer.weight_momentums= self.beta_1 * layer.weight_momentums + ( 1-self.beta_1 ) * layer.dweights
        layer.bias_momentums= self.beta_1 * layer.bias_momentums + ( 1-self.beta_1 ) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / ( 1 - self.beta_1 ** (self.iterations + 1 ))
        bias_momentums_corrected = layer.bias_momentums / ( 1 - self.beta_1 ** (self.iterations + 1 ))
        # Update cache with squared current gradients
        layer.weight_cache= self.beta_2 * layer.weight_cache + ( 1-self.beta_2 ) * layer.dweights**2
        layer.bias_cache= self.beta_2 * layer.bias_cache  + ( 1-self.beta_2) * layer.dbiases**2
        # Get corrected momentum
        weight_cache_corrected = layer.weight_cache / ( 1 - self.beta_2 ** (self.iterations + 1 ))
        bias_cache_corrected = layer.bias_cache / ( 1 - self.beta_2 ** (self.iterations + 1 ))
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += - self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected)+self.epsilon)
        layer.biases += - self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected)+self.epsilon)

    # Call once after any parameter updates
    def post_update_params (self):
        self.iterations += 1


# Create dataset
X, y = sine_data()
# Create Dense layer with 1 input feature and 64 output values
dense1 = Layer_Dense (1, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense (64, 64)
activation2 = Activation_ReLU()
dense3 = Layer_Dense (64, 1)
activation3= Activation_Linear()
# Create loss function
loss_function = Loss_MeanSquaredError ()
# Create optimizer
optimizer = Optimizer_Adam(learning_rate = 0.05 , decay = 1e-3)
# Accuracy precision for accuracy calculation
# There are no really accuracy factor for regression problem,
# but we can simulate/approximate it. We'll calculate it by checking
# how many values have a difference to their ground truth equivalent
# less than given precision
# We'll calculate this precision as a fraction of standard deviation
# of al the ground truth values
accuracy_precision = np.std(y) / 250

# Train in loop
for epoch in range ( 10001 ):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    # Calculate the data loss
    data_loss = loss_function.calculate(activation3.output, y)
    # Calculate regularization penalty
    regularization_loss = loss_function.regularization_loss (dense1) + loss_function.regularization_loss(dense2) + loss_function.regularization_loss(dense3)
    # Calculate overall loss
    loss = data_loss + regularization_loss
    # Calculate accuracy from output of activation2 and targets
    # To calculate it we're taking absolute difference between
    # predictions and ground truth values and compare if differences
    # are lower than given precision value
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions -y) < accuracy_precision)
    if not epoch % 100:
        print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f},' + f'loss: {loss:.3f} (' + f'data_loss: {data_loss: .3f},' + f'reg_loss: {regularization_loss:.3f}),' + f'lr: {optimizer.current_learning_rate}')
    # Backward pass
    loss_function.backward(activation3.output, y) 
    activation3.backward(loss_function.dinputs) 
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs) 
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer. update_params (dense1) 
    optimizer. update_params(dense2)
    optimizer. update_params(dense3) 
    optimizer.post_update_params()
x_test, y_test = sine_data()
dense1.forward(x_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)
plt.plot(x_test, y_test)
plt.plot(x_test, activation3.output)
plt.show()
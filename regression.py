import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data
import numpy as np 

# Model class
class Model:
    def __init__(self):
        # Create a list of network objects
        self.layers = []
    # Add objects to the model
    def add (self, layer):
        self.layers.append (layer)
    
    # Set loss and optimizer
    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    # Finalize the model
    def finalize (self):
        # Create and set the input layer
        self.input_layer = Layer_Input ()
        # Count all the objects
        layer_count = len (self.layers)
        # Initialize a list containing trainable layers:
        self.trainable_layers = []
        # Iterate the objects
        for i in range (layer_count):
            # If it's the first layer,
            # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            # The last layer - the next object is the loss
            else:
                self.layers[i].prev = self.layers [i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            # If layer contains an attribute called "weights"
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            if hasattr (self.layers[i],'weights' ):
                self.trainable_layers.append(self.layers[i])
        
    #Train the model
    def train ( self , X , y ,*, epochs = 1 , print_every = 1 ):
        # Main training loop
        for epoch in range ( 1 , epochs + 1 ):
            # Perform the forward pass
            output = self.forward(X)
            # Temporary
            print (output)
            exit()
    
    # Performs forward pass
    def forward (self, X):
        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X)
        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output)
        # "layer" is now the last object from the list,
        # return its output
        return layer.output

# Input "layer"
class Layer_Input :
    # Forward pass
    def forward ( self , inputs ):
        self.output = inputs


# Dense layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons,weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
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
    # Set/remember trainable layers
    def remember_trainable_layers (self, trainable_layers):
        self.trainable_layers = trainable_layers
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate (self, output, y) :
        # Calculate sample losses
        sample_losses = self.forward (output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss, self.regularization_loss()
    # Regularization loss calculation
    def regularization_loss ( self , layer ):
        # 0 by default
        regularization_loss = 0

        
        for layer in trainable_layers:

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
# Instantiate the model
model = Model()
# Add layers
model.add(Layer_Dense (1, 64)) 
model.add(Activation_ReLU()) 
model.add(Layer_Dense (64, 64)) 
model.add(Activation_ReLU())
model.add(Layer_Dense (64, 1)) 
model.add(Activation_Linear())

# Set loss and optimizer objects
model.set(loss = Loss_MeanSquaredError(),optimizer = Optimizer_Adam( learning_rate = 0.005 , decay = 1e-3 ),)

# Finalize the model
model.finalize()
# Train the model
model.train(X, y, epochs = 10000 , print_every = 100 )
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# activation functions are in every node and are actived after the input*weights + bias calc
# used to introduce non-linearity into the model which allows it to "fit" more complex data
# output of the activation function is then sent to the next layer

# Common activation functions are as follows:

# Sigmoid: y = 1/(1 + e^(-x))
# generally obsolete because of the vanishing gradient problem

#ReLU: y = max(0, x)  (rectified linear unit)
# reLu used to avoid the vanishing gradient problem with sigmoid, and it's very fast

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class LayerDense:
    def __init__(self, n_inputs, n_neurons): 
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1, n_neurons)) 
    def forward(self, inputs):
        self.output = np.matmul(inputs, self.weights) + self.biases

# Softmax: e^(xi) / sum(e^(xi)) for all xi in the layer
# usually used in the output layer of a classification problem
# useful because it allows the model to "relate" the outputs of the neurons to each other
# also good because it normalizes the output so that the sum of the outputs is 1, and kind of creates a probability distribution with the outputs
# this is useful for classification problems because it allows the model to "choose" the most likely class

class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True) # axis=1 means sum across the rows, keepdims=True means keep the shape of the array the same


X, y = spiral_data(100, 3) # y usually used to denote the "target" data



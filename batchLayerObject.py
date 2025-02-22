# Batches: why?
# Allows for parallel processing, and helps to generalize by  reducing erratic movement in a "fitment" line
# don't want to input the whole dataset at once because it will overfit, which is bad for generalization
# as a result batch sizes are usually between 8 and 128

import numpy as np

np.random.seed(0) # to get the same random numbers each time

inputs = [[1,2,3,2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]  # a batch of 3 inputs
# after this point inputs will be X as is common practice

weights1 = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]] # kind of a matrix
biases1 = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

layer1outputs = np.matmul(inputs, np.array(weights1).T) + biases1 # T is the transpose operator in numpy
layer2outputs = np.matmul(layer1outputs, np.array(weights2).T) + biases2 

# to make this more general, we can define a class for a layer

class LayerDense:
    def __init__(self, n_inputs, n_neurons): # want to initialise weights as somewhere between 0 and 1, biases usually all 0
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons) #randn in a gaussian distribution around 0
        # the shape of the weights matrix is n_inputs x n_neurons so we don't have to transpose every time
        self.biases = np.zeros((1, n_neurons)) # np.zeros sets all to zero, note shape is passed as the first input of zeros so need a tuple in a tuple here
    def forward(self, inputs):
        self.output = np.matmul(inputs, self.weights) + self.biases



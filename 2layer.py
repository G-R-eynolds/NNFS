import numpy as np

inputs = [1,2,3,2.5]  #4 inputs means prev layer is 4 neurons
weights1 = [0.2, 0.8, -0.5, 1.0] # 3  weights , 1 for each input
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
bias1 = 2
bias2 = 3
bias3 = 0.5

output = [np.dot(weights1, inputs) + bias1, np.dot(weights2, inputs) + bias2, np.dot(weights3, inputs) + bias3]  # output is then sent to all neurons in the next layer
# this essentially models a layer of 3 neurons with 4 inputs (4 neurons in the previous layer)
# in a fully connected NN all outputs are sent to every neuron in the next layer, so can model the output as a vector

print(output) 
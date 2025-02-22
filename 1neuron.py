import numpy as np

inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7] # 3 inputs, 3 weights
bias = 3

output = 0
for i in range (0,3):
    output += inputs[i] * weights[i] 
output += bias                           # the v basic code for a neuron

output = np.dot(inputs, weights) + bias  # simpler with a dot product

print(output)
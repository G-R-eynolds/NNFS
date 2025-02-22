import numpy as np

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]] # kind of a matrix
biases = [2, 3, 0.5]
inputs = [1, 2, 3, 2.5]

output1 = np.dot(weights, inputs) + biases  # feels like abuse of notation but this does actually work
output = np.matmul(weights, inputs) + biases  # fast  yay
print(output1)
print(output)
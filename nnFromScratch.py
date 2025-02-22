import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
X, y = spiral_data(100, 3) 

class LayerDense:
    def __init__(self, n_inputs, n_neurons): 
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1, n_neurons)) 

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.matmul(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.matmul(self.inputs.T, dvalues)  # dvalues is the gradient of the loss function with respect to the output of the layer
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) 
        self.dinputs = np.matmul(dvalues, self.weights.T) 

class ActivationReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0  # if the input was less than or equal to 0, the gradient is 0

class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class CCE(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            confidences = clipped[range(samples), y_true]
            losses = -np.log(confidences)
        elif len(y_true.shape) == 2:
            confidences = np.sum(clipped*y_true, axis=1)
            losses = -np.log(confidences)
        return losses

# now that we are implementing a backward pass, is faster to combine Softmax and CCE into one class:

class SoftmaxCCE():
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = CCE()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):   # compputes dinputs, using the combined class simplifies the calculation to dinputs = (probabilities - y_true) / number of samples
        samples = len(y_true)
        if len(y_true.shape) == 2:  # turn into scalar if one-hot encoded
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -=1 
        self.dinputs = self.dinputs / samples

class OptimizerSGD:
    def __init__(self, learning_rate = 1.0, decay = 0, momentum = 0.99):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.momentum = momentum
        self.iterations = 0
        self.decay = decay

    def pre_update(self, layer):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):  # if the layer does not have the momentum attributes, create them
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        
        else:
            weight_updates = self.learning_rate * layer.dweights
            bias_updates = self.learning_rate * layer.dbiases
               
        layer.weights = weight_updates
        layer.biases = bias_updates

    def post_update(self):
        self.iterations += 1

class OptimizerAdam:
    def __init__(self, learning_rate = 0.001, decay = 0, epsilon = 1e-7, beta1 = 0.9, beta2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))
    
    def update(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_velocities = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_velocities = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dweights
        layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.dbiases

        # Mt calclation
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1 ** (self.iterations + 1))

        # Update velocities with squared current gradients (Vt calculation)
        layer.weight_velocities = self.beta2 * layer.weight_velocities + (1 - self.beta2) * layer.dweights ** 2
        layer.bias_velocities = self.beta2 * layer.bias_velocities + (1 - self.beta2) * layer.dbiases ** 2

        # Get corrected Vt
        weight_velocities_corrected = layer.weight_velocities / (1 - self.beta2 ** (self.iterations + 1))
        bias_velocities_corrected = layer.bias_velocities / (1 - self.beta2 ** (self.iterations + 1))

        # Update
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_velocities_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_velocities_corrected) + self.epsilon)

    def post_update(self):
        self.iterations += 1



layer1 = LayerDense(2, 64)
activation1 = ActivationReLU()
layer2 = LayerDense(64, 64)
activation2 = ActivationReLU()
layer3 = LayerDense(64, 3)
loss_activation = SoftmaxCCE()  # loss implemented here


epochs = 10001

optim = OptimizerAdam(learning_rate=0.001, decay=5e-7)

def train(X, y):
    for epoch in range(0, epochs):
        layer1.forward(X)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)
        layer3.forward(activation2.output)
        loss = loss_activation.forward(layer3.output, y)

        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions==y)

        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}')

        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        layer3.backward(loss_activation.dinputs)
        activation2.backward(layer3.dinputs)
        layer2.backward(activation2.dinputs)
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)

        # Update weights and biases
        optim.pre_update()
        optim.update(layer1)
        optim.update(layer2)
        optim.post_update()

train(X, y)

X_test, y_test = spiral_data(100, 3)

layer1.forward(X_test)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
layer3.forward(activation2.output)
loss = loss_activation.forward(layer3.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)

if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

import numpy as np

# generally when building a classifier network with softmax output, the ollloss function is Catgorical Cross-Entropy:
# L_i = -log(y_hat_i,k) where y_hat is the predicted probability of the correct class
# this loss function turns out to be very convenient for backpropagation

class Loss:
    def calculate(self, output, y):   # y are intended target values
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):  # y_pred is "output" in the Loss class, y_tru is y
        samples = len(y_pred)
        clipped = np.clip(y_pred, 1e-7, 1-1e-7) # to avoid log(0) which is undefined

        if len(y_true.shape) == 1:  # i.e y values are being passed as scalars
            confidences = clipped[range(samples), y_true]   # this is a numpy trick that multiplies the outputs by the correct class as a 1-hot encoded vector
            losses = -np.log(confidences)

        elif len(y_true.shape) == 2:  # i.e y values are being passed as one-hot encoded vectors
            confidences = np.sum(clipped*y_true, axis=1) # use axis=1 to sum across the rows
            losses = -np.log(confidences)

        return losses


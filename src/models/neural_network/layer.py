import numpy as np
import random

from src.models.neural_network.activations import (relu, relu_backward,
                                                   softmax)


class DenseLayer:
    def __init__(self, layer_size: int,
                 activation: str,
                 random_state=None):

        self.layer_size = layer_size
        self.activation = activation
        self.random_state = random_state

        if self.random_state:
            np.random.seed(random_state)
            random.seed(random_state)

        self.activation_function = self._get_activation_function()

    def forward(self, inputs, weights, bias):
        z = np.dot(inputs, weights.T) + bias
        activation = self.activation_function(z)
        return activation, z

    def backward(self, dA_curr, W_curr, Z_curr, A_prev):
        if self.activation == 'softmax':
            dW = np.dot(A_prev.T, dA_curr)
            db = np.sum(dA_curr, axis=0, keepdims=True)
            dA = np.dot(dA_curr, W_curr)
        elif self.activation == "relu":
            dZ = relu_backward(dA_curr, Z_curr)
            dW = np.dot(A_prev.T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)
            dA = np.dot(dZ, W_curr)

        else:
            raise ValueError("Provided activation_function is not implemented")

        return dA, dW, db

    def _get_activation_function(self):
        if self.activation == "relu":
            activation_function = relu
        elif self.activation == "softmax":
            activation_function = softmax
        else:
            raise ValueError("Provided activation_function is not implemented")
        return activation_function

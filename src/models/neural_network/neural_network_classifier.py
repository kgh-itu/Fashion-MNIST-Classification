from __future__ import annotations

from sklearn.utils import shuffle  # I assume we can use this from sk-learn

import numpy as np
from typing import Union
from sklearn.model_selection import train_test_split

from src.models.neural_network.layer import DenseLayer
from src.get_train_test_split.fashion_mnist_data import FashionMnistData


from src.models.neural_network.loss import (delta_cross_entropy,
                                            get_accuracy,
                                            calculate_loss)


class NeuralNetworkClassifier:
    def __init__(self,
                 layers: list[DenseLayer],
                 learning_rate: Union[float | int],
                 epochs: int):

        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.trainable_params: list[dict] = []
        self.architecture: list[dict] = []
        self.cache: list[dict] = []
        self.derivatives: list[dict] = []

        self.history = {"epochs": [i + 1 for i in range(self.epochs)], "train_loss": [], "train_accuracy": [],
                        "validation_loss": [], "validation_accuracy": []}

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            batch_size: int,
            validation_size: Union[float | int] = 0,
            random_state: int = 42) -> dict:

        self._configure_neural_network(x_train)
        self._init_trainable_params()

        num_batches = int(np.ceil(x_train.shape[0] / batch_size))

        if validation_size:
            x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train,
                                                                            test_size=validation_size,
                                                                            random_state=random_state)

        for i in range(1, self.epochs + 1):
            x_train, y_train = shuffle(x_train, y_train)
            x_batches = np.array_split(x_train, num_batches)
            y_batches = np.array_split(y_train, num_batches)
            for x_batch, y_batch in zip(x_batches, y_batches):  # mini batch gradient descent
                y_pred = self._forward(x_batch)
                loss = delta_cross_entropy(predicted=y_pred, y_true=y_batch)
                self._backward(loss)
                self._update_trainable_params()

            acc, loss = self._get_performance_after_epoch(x_train, y_train)
            self.history["train_accuracy"].append(acc)
            self.history["train_loss"].append(loss)

            print(f"Epoch {i}: Train Accuracy: {round(acc, 3)} Train Loss {round(loss, 3)}")

            if validation_size:
                acc, loss = self._get_performance_after_epoch(x_validation, y_validation)
                self.history["validation_accuracy"].append(acc)
                self.history["validation_loss"].append(loss)

        return self.history

    def predict(self, X):
        y_pred = self._forward(X)
        return np.argmax(y_pred, axis=1)

    def _forward(self, input_data):
        # activations_prev is input to the current layer
        # z_curr is the output of the current layer before an activation has been applied

        self.cache = []  # clear the cache

        activations_current = input_data
        for i in range(len(self.trainable_params)):
            activations_prev = activations_current
            activations_current, z_curr = self.layers[i].forward(inputs=activations_prev,
                                                                 weights=self.trainable_params[i]['W'],
                                                                 bias=self.trainable_params[i]['b'])

            self.cache.append({'inputs': activations_prev, 'Z': z_curr})  # needed for backwards pass

        return activations_current

    def _backward(self, error):
        dA_prev = error
        self.derivatives = []

        for idx, layer in reversed(list(enumerate(self.layers))):
            dA_curr = dA_prev
            A_prev = self.cache[idx]['inputs']
            Z_curr = self.cache[idx]['Z']
            W_curr = self.trainable_params[idx]['W']
            dA_prev, dW_curr, db_curr = layer.backward(dA_curr, W_curr, Z_curr, A_prev)

            self.derivatives.append({'dW': dW_curr, 'db': db_curr})

        self.derivatives = list(reversed(self.derivatives))  # reverse derivatives to match order.

    def _update_trainable_params(self):
        for idx, layer in enumerate(reversed(self.layers)):
            self.trainable_params[idx]['W'] -= self.learning_rate * self.derivatives[idx]['dW'].T
            self.trainable_params[idx]['b'] -= self.learning_rate * self.derivatives[idx]['db']

    def _init_trainable_params(self):
        self.trainable_params = [{
            'W': np.random.randn(layer['output_dim'], layer['input_dim']) * np.sqrt(2 / layer['input_dim']),
            'b': np.zeros((1, layer['output_dim']))
        } for layer in self.architecture]

    def _configure_neural_network(self, input_data):
        """Sets up the correct dimensions and architecture based on the input data,
        and the input layers"""

        def build_layer(input_dim, output_dim, activation):
            return {'input_dim': input_dim,
                    'output_dim': output_dim,
                    'activation': activation}

        self.architecture.append(
            build_layer(input_data.shape[1],
                        self.layers[0].layer_size,
                        self.layers[0].activation)
        )

        for i in range(1, len(self.layers)):
            self.architecture.append(
                build_layer(self.layers[i - 1].layer_size,
                            self.layers[i].layer_size,
                            self.layers[i].activation)
            )

    def add(self, layer):
        self.layers.append(layer)

    def _get_performance_after_epoch(self, x_train, y_true):
        y_pred = self._forward(x_train)
        loss = calculate_loss(predicted=y_pred, y_true=y_true)
        acc = get_accuracy(predicted=y_pred, y_true=y_true)
        return acc, loss


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    ep = 20
    lr = 0.01
    data = FashionMnistData()
    x_t, y_t, _, _ = data.get_train_test_split(normalize=True)

    our_model = NeuralNetworkClassifier(layers=[DenseLayer(128, "relu"),
                                                DenseLayer(64, "relu"),
                                                DenseLayer(5, "softmax")],
                                        learning_rate=lr,
                                        epochs=ep)

    our_model.fit(x_t, y_t, batch_size=32, validation_size=0.5)

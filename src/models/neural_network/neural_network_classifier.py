from sklearn.utils import shuffle  # I assume we can use this from sk-learn
import warnings
import numpy as np


from src.models.neural_network.layer import DenseLayer
from src.get_train_test_split.fashion_mnist_data import FashionMnistData

from src.models.neural_network.loss import (delta_cross_entropy,
                                            get_accuracy,
                                            calculate_loss)


class NeuralNetworkClassifier:
    def __init__(self,
                 layers: list[DenseLayer],
                 learning_rate: float,
                 epochs: int):

        self.layers = layers

        if len(self.layers) == 0:
            warnings.warn("Remember to manually add layers with self.add()")

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.trainable_params: list[dict] = []
        self.architecture: list[dict] = []
        self.cache: list[dict] = []
        self.derivatives: list[dict] = []

    def fit(self, X_train, y_train, batch_size):
        self._configure_neural_network(X_train)
        self._init_trainable_params()

        num_batches = int(np.ceil(X_train.shape[0] / batch_size))

        for i in range(self.epochs):
            # Shuffle the training data at the beginning of each epoch
            X_train, y_train = shuffle(X_train, y_train)
            # Split the training data into mini-batches
            X_batches = np.array_split(X_train, num_batches)
            y_batches = np.array_split(y_train, num_batches)
            # Loop over the mini-batches
            for X_batch, y_batch in zip(X_batches, y_batches):
                y_pred = self._forward(X_batch)
                loss = delta_cross_entropy(predicted=y_pred, y_true=y_batch)
                self._backward(loss)
                self._update_trainable_params()

            # Calculate and print the loss and accuracy for the entire training set after each epoch
            y_pred = self._forward(X_train)
            prediction_loss = calculate_loss(predicted=y_pred, y_true=y_train)
            prediction_accuracy = get_accuracy(predicted=y_pred, y_true=y_train)

            if i % 20 == 0:
                print(f"EPOCH {i}: TRAIN ACCURACY {prediction_accuracy} TRAIN LOSS {prediction_loss}")

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

    def _configure_neural_network(self, data):
        """Sets up the correct dimensions and architecture based on the input data,
        and the input layers"""

        def build_layer(input_dim, output_dim, activation):
            return {'input_dim': input_dim,
                    'output_dim': output_dim,
                    'activation': activation}

        self.architecture.append(
            build_layer(data.shape[1],
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


if __name__ == "__main__":
    data_ = FashionMnistData(r"C:\Users\Bjarne\Desktop\Fashion-MNIST-Classification\data")
    X_train_, y_train_, X_test_, y_test_ = data_.get_train_test_split(normalize=True)
    model = NeuralNetworkClassifier(layers=[],
                                    learning_rate=0.1,
                                    epochs=20)

    model.add(DenseLayer(120, "relu"))
    model.add(DenseLayer(60, "relu"))
    model.add(DenseLayer(5, "softmax"))
    model.fit(X_train_, y_train_, batch_size=10)

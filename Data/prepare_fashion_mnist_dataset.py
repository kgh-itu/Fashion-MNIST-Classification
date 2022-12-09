import numpy as np


class FashionMnistData:
    def __init__(self, get_train=True, get_test=True):
        self.get_train = get_train
        self.get_test = get_test
        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

        if self.get_train:
            self.train = np.load("raw_data/fashion_train.npy")
            self.x_train = self.train[:, : -1]
            self.y_train = self.train[:, -1]

        if self.get_test:
            self.test = np.load("raw_data/fashion_test.npy")
            self.x_test = self.test[:, : -1]
            self.y_test = self.test[:, -1]

    def get(self):
        return self.x_train, self.y_train, self.x_test, self.y_test


import numpy as np


class FashionMnistData:
    def __init__(self, get_train=True, get_test=True, path_to_folder="raw_data", normalize=False):
        self.get_train = get_train
        self.get_test = get_test
        self.normalize = normalize
        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

        if self.get_train:
            self.train = np.load(f"{path_to_folder}/fashion_train.npy")
            self.x_train = self.train[:, : -1]
            self.y_train = self.train[:, -1]

        if self.get_test:
            self.test = np.load(f"{path_to_folder}/fashion_test.npy")
            self.x_test = self.test[:, : -1]
            self.y_test = self.test[:, -1]

    def get(self):
        x_train, y_train, x_test, y_test = self.x_train, self.y_train, self.x_test, self.y_test

        if self.normalize:
            x_train, x_test = x_train / 255, x_test / 255

        return x_train, y_train, x_test, y_test


import numpy as np


class FashionMnistData:
    def __init__(self, get_train=True, get_test=True, original_data_set=False,
                 path_to_folder="data", normalize=False):

        self.get_train = get_train
        self.get_test = get_test
        self.normalize = normalize # dividing the pixels by 255
        self.original_data_set = original_data_set

        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None

        if self.get_train:
            self.train = np.load(f"{path_to_folder}/fashion_train.npy")
            self.X_train = self.train[:, : -1]
            self.y_train = self.train[:, -1]

        if self.get_test:
            self.test = np.load(f"{path_to_folder}/fashion_test.npy")
            self.X_test = self.test[:, : -1]
            self.y_test = self.test[:, -1]

    def get(self):
        if self.original_data_set:
            return self.train, self.test

        X_train, y_train, X_test, y_test = self.X_train, self.y_train, self.X_test, self.y_test

        if self.normalize:
            if self.get_test:
                X_train, X_test = X_train / 255, X_test / 255
            else:
                X_train = X_train / 255
                
        return X_train, y_train, X_test, y_test

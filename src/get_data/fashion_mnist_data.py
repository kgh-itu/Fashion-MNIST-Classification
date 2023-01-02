import numpy as np


class FashionMnistData:
    def __init__(self,
                 path_to_folder="data",
                 normalize=True):

        self.train = np.load(f"{path_to_folder}/fashion_train.npy")
        self.test = np.load(f"{path_to_folder}/fashion_test.npy")

        # TRAINING DATA
        self.X_train = self.train[:, : -1].astype(float)
        self.y_train = self.train[:, -1].astype(int)

        # TEST DATA
        self.X_test = self.test[:, : -1].astype(float)
        self.y_test = self.test[:, -1].astype(int)

        self.normalize = normalize

    def get_train_test_split(self):
        if self.normalize:
            self.X_train /= 255
            self.X_test /= 255

        return self.X_train, self.y_train, self.X_test, self.y_test

    def get_original_datasets(self):
        return self.train, self.test

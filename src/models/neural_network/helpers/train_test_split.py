import numpy as np


def train_test_split(X, y, validation_size, random_state=42):
    if random_state:
        np.random.seed(random_state)
    indices = np.random.permutation(X.shape[0])
    split = int(X.shape[0] * (1 - validation_size))
    train_indices = indices[:split]
    test_indices = indices[split:]
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_train, y_train, X_test, y_test

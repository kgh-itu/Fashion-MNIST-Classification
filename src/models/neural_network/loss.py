import numpy as np


def get_accuracy(predicted, y_true):
    return np.mean(np.argmax(predicted, axis=1) == y_true)


def calculate_loss(predicted, y_true):
    predicted = predicted.copy()
    y_true = y_true.copy()
    num_samples = y_true.shape[0]

    correct_log_probs = -np.log(predicted[range(num_samples), y_true])
    return np.sum(correct_log_probs) / num_samples


# "https://deepnotes.io/softmax-crossentropy"
def delta_cross_entropy(predicted, y_true):
    grad = predicted.copy()
    y_true = y_true.copy()
    num_samples = y_true.shape[0]

    grad[range(num_samples), y_true] -= 1
    return grad / num_samples

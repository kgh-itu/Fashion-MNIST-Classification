import numpy as np


def activation_function(activation_str):
    if activation_str == "relu":
        return relu
    elif activation_str == "softmax":
        return softmax


def relu(x):
    return np.maximum(0, x)


def relu_backward(d_A, Z_out):
    """derivative of relu w.r.t input"""
    d_Z = np.array(d_A, copy=True)
    d_Z[Z_out <= 0] = 0
    return d_Z


def softmax(y_preds):
    exp_y = np.exp(y_preds)
    return exp_y / exp_y.sum(axis=1, keepdims=True)


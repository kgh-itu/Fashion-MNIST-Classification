import numpy as np


def calculate_gini(left_y, right_y):
    """
    Calculates the total gini impurity of a split.

    Parameters
    ----------
    left_y : np.ndarray
        The labels of the left split.
    right_y : np.ndarray
        The labels of the right split.

    Returns
    -------
    float
        The gini impurity of the split."""

    left_gini = get_node_gini(left_y)
    right_gini = get_node_gini(right_y)

    left_weight = len(left_y) / (len(left_y) + len(right_y))
    right_weight = len(right_y) / (len(right_y) + len(left_y))

    return left_gini * left_weight + right_gini * right_weight


def get_node_gini(y):
    """
        Calculates the gini impurity of a node.

        Parameters
        ----------
        y : np.ndarray
            The labels of the node.

        Returns
        -------
        float
            The gini impurity of the node."""

    labels, count = np.unique(y, return_counts=True)
    probs_squared = (count / len(y)) ** 2
    return 1 - probs_squared.sum()


def calculate_entropy(left_y, right_y):
    """
    Calculates the total entropy of a split.

    Parameters
    ----------
    left_y : np.ndarray
        The labels of the left split.
    right_y : np.ndarray
        The labels of the right split.

    Returns
    -------
    float
        The entropy of the split.
    """
    left_entropy = get_node_entropy(left_y)
    right_entropy = get_node_entropy(right_y)

    left_weight = len(left_y) / (len(left_y) + len(right_y))
    right_weight = len(right_y) / (len(right_y) + len(left_y))

    return left_entropy * left_weight + right_entropy * right_weight


def get_node_entropy(y):
    """
    Calculates the entropy of a node.

    Parameters
    ----------
    y : np.ndarray
        The labels of the node.

    Returns
    -------
    float
        The entropy of the node.
    """
    labels, count = np.unique(y, return_counts=True)
    probs = count / len(y)
    return -(probs * np.log2(probs)).sum()

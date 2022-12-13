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


def get_possible_cutoffs(x):
    """
        Gets all possible cutoffs for a feature with following steps:
        1) Get unique values for feature
        2) Sort unique values
        3) Calculate moving average with a window of 2 like such:
            _moving_average([1, 2, 3, 4]) -> [1.5, 2.5, 3.5]

        Parameters
        ----------
        x : np.ndarray
            The current feature array

        Returns
        -------
        ndarray[float]
            cutoffs for x."""

    x = np.sort(np.unique(x))
    moving_avg = _moving_average(x)
    return moving_avg


def _moving_average(x: np.ndarray, window=2):
    """
        Calculates moving average of an array

        Parameters
        ----------
        x : np.ndarray
            The current feature array

        Returns
        -------
        ndarray[float]
            moving average of x."""

    return np.convolve(x, np.ones(window), 'valid') / window
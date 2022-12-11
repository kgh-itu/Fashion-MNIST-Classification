import numpy as np


def calculate_gini(left_y, right_y) -> float:
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


def get_node_gini(y: np.ndarray) -> float:
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


def get_possible_cutoffs(x: np.ndarray):
    """
        Calculates the gini impurity of a node.

        Parameters
        ----------
        x : np.ndarray
            The current feature

        Returns
        -------
        ndarray[float]
            All possible cutoffs for x."""

    x = np.sort(np.unique(x))
    moving_avg = _moving_average(x)
    return moving_avg


def _moving_average(x: np.ndarray) -> np.ndarray:
    window = 2
    return np.convolve(x, np.ones(window), 'valid') / window


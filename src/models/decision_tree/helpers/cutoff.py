import numpy as np


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

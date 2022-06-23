import numpy as np


def linreg(X: np.ndarray, w: np.ndarray, b: np.ndarray):
    """Compute predictions based on multiple linear regression model.

    Parameters
    ----------
    X : array_like
        Input 2-D array with `m` examples and `n` features.
    w : array_like
        Parameters, 1-D array with the same number of features `n`. 
    b : float
        Bias parameter.
    """
    return X @ w + b

import numpy as np

from ..models import predictions


def mse(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    """Compute gradient of mean squared error.

    Parameters
    ----------
    X : array_like
        Input 2-D array with `m` examples and `n` features.
    y : array_like
        Output 1-D array with `m` labels.
    w : array_like
        Parameters, 1-D array with the same number of features `n`. 
    b : float
        Bias parameter.

    Returns
    -------
    array_like
        Gradient of mean squared error.
    """
    m, n = X.shape

    X_full = np.column_stack((np.ones(m), X))
    w_full = np.concatenate(([b], w))

    gradient = 1/m * X_full.T @ (X_full @ w_full - y)

    return gradient

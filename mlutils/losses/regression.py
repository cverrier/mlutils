import numpy as np

from ..models import predictions


def mse(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    """Compute mean squared error for multiple linear regression.

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
    """
    m = X.shape[0]

    y_pred = predictions.linreg(X, w, b)
    cost = 1/(2*m) * np.sum((y_pred - y) ** 2)

    return cost

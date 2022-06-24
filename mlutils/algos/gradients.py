import numpy as np

from ..models import predictions


def gradient_descent(X: np.ndarray, y: np.ndarray, w_init: np.ndarray,
                     b_init: float = None, cost_func: function, grad_func: function,
                     alpha: float, n_iters: int = 1000):
    if b_init is not None:
        w = np.concatenate(([b_init], w_init))
    else:
        w = w_init.copy()

    loss_hist = []

    for i in range(n_iters):
        grad = grad_func(X, y, w)

        w -= alpha * grad

        if i % 10 == 0:
            # TODO: Incorporate bias term in `w` in cost function and
            # also in predictions.
            loss_hist.append(cost_func(X, y, w))


def mse(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float = None):
    """Compute gradient of mean squared error.

    Parameters
    ----------
    X : array_like
        Input 2-D array with `m` examples and `n` features.
    y : array_like
        Output 1-D array with `m` labels.
    w : array_like
        Parameters, 1-D array with the same number of features `n`. 
    b : float, optional
        Bias parameter.

    Returns
    -------
    array_like
        Gradient of mean squared error.
    """
    m, n = X.shape

    X_full = np.column_stack((np.ones(m), X))

    if b is not None:
        w_full = np.concatenate(([b], w))
    else:
        w_full = w.copy()

    gradient = 1/m * X_full.T @ (X_full @ w_full - y)

    return gradient

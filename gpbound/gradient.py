from typing import Callable
import numpy as np
from scipy.optimize import approx_fprime


def estimate_c(X: np.ndarray, f: Callable, h: float = None) -> np.ndarray:
    """Estimate C

    C is a (D, D) matrix C_kl = E[f_k f_l] where f_k is the partial derivative of f
    with respect to the dimension x_k.

    The currently implemented gradient approximation methods return a diagonal C,
    with C_kk = E[f_k^2].

    Args:
        X: Data as (N, D) array.
        y: Target as (N, 1) array.
        f: Regressor. See estimate_gradient_fd().
        h: Finite-difference step for gradient approximation.
        Defaults to `sqrt(np.finfo(float).eps)`, which is approximately 1.49e-08.

    Returns:
        C: (D, D) array
    """
    h = np.sqrt(np.finfo(float).eps) if h is None else h
    g = estimate_gradient_fd(X=X, f=f, h=h)
    C = (g**2).mean(axis=0)
    return np.diag(C)


def estimate_gradient_fd(X: np.ndarray, f: Callable, h: float = None, vectorized=True):
    """Estimate the gradient of a regressor using (forward) finite differences

    Args:
        X: Data as (N, D) array.
        f: Regressor. f(x) should work for x as a (D,) array or a (1, D) array.
        h: Finite difference step.
        vectorized: If f is vectorized.
        - Set to True, if f(X) works for X as a (N, D) array and returns a (N,) or (N, 1) array.
        - Set to False, if f(x) works for x as a (D,) or (1, D) array and returns a scalar.

    Returns:
        Approximate gradient of f at every point in X as (N, D) array.
    """
    N, D = X.shape
    h = np.sqrt(np.finfo(float).eps) if h is None else h

    if vectorized:
        I = np.eye(D)
        df = []
        for j in range(D):
            df.append((f(X + I[j] * h) - f(X)).reshape(N, 1))
        return np.hstack(df) / h
    else:
        _f = f
        try:
            _ = _f(X[0, :])
        except:
            _f = lambda x: f(x.reshape(1, -1))
        g = []
        for i in range(N):
            g_i = approx_fprime(xk=X[i, :], f=_f, epsilon=h)
            g.append(g_i)
        return np.vstack(g)

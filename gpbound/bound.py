import numpy as np
from functools import cached_property
from scipy.spatial.distance import cdist
from .gradient import estimate_c
from sklearn.ensemble import RandomForestRegressor


class GPBound:
    def __init__(
        self,
        C: np.ndarray = None,
        sigma0_sq: float = 1.0,
        sigma_sq: float = 0.0,
    ):
        """GP Bound

        Args:
            C: Expected squared gradient as (D, D) array. Defaults to None.
            sigma0_sq: Signal variance. Defaults to None.
            sigma_sq: Noise variance. Defaults to 0.0.
        """
        self.C = C
        self.sigma0_sq = sigma0_sq
        self.sigma_sq = sigma_sq

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        self.X = np.atleast_2d(X)
        self.y = np.atleast_1d(y)
        if self.C is None:
            fhat = RandomForestRegressor(random_state=0).fit(X, y.reshape(-1))
            self.C = estimate_c(X=X, f=fhat.predict, h=0.01)
        return self

    def var(self, X_new: np.ndarray):
        X_new = np.atleast_2d(X_new)
        return var_bound_scaled(
            Z_new=X_new @ self.L,
            Z=self.X @ self.L,
            sigma0_sq=self.sigma0_sq,
            sigma_sq=self.sigma_sq,
        )

    def bias(self, X_new: np.ndarray, Yhat_new: np.ndarray, version: str = "exact"):
        X_new = np.atleast_2d(X_new)
        Yhat_new = np.atleast_1d(Yhat_new)
        return bias_bound_scaled(
            Z_new=X_new @ self.L,
            Yhat_new=Yhat_new,
            Z=self.X @ self.L,
            y=self.y,
            sigma0_sq=self.sigma0_sq,
            sigma_sq=self.sigma_sq,
            version=version,
        )

    @cached_property
    def L(self):
        return np.linalg.cholesky(self.C)


def var_bound(
    X_new: np.ndarray,
    X: np.ndarray,
    C: np.ndarray,
    sigma0_sq: float,
    sigma_sq: float = 0,
):
    """Posterior variance bound

    Args:
        X_new: (M, D) array of test points on which to evaluate the variance bound.
        X: (N, D) array of train points.
        C: (D, D) array of gradient covariances.
        sigma0_sq: Signal variance.
        sigma_sq: Noise variance. Defaults to 0.

    Returns:
        (M,) array of the posterior variance bound at the test points in X_new.
    """
    L = np.linalg.cholesky(C)
    return var_bound_scaled(
        Z_new=X_new @ L, Z=X @ L, sigma0_sq=sigma0_sq, sigma_sq=sigma_sq
    )


def var_bound_scaled(
    Z_new: np.ndarray,
    Z: np.ndarray,
    sigma0_sq: float = 1,
    sigma_sq: float = 0,
):
    """Posterior variance bound when C is the identity matrix

    See var_bound().
    """
    # d: (N * N_new) array, d_ij = \sqrt{\sum_k (Z[i,k] - Z_new[j,k])^2} / sigma0
    d = cdist(Z, Z_new) / sigma0_sq**0.5
    V = (sigma0_sq**2 * sin_(d) ** 2 + sigma0_sq * sigma_sq) / (sigma0_sq + sigma_sq)
    return np.min(V, axis=0)


def bias_bound(
    X_new: np.ndarray,
    Yhat_new: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    sigma0_sq: float,
    sigma_sq: float,
    version="exact",
):
    """Bias bound

    Args:
        X_new: (M, D) array of test points on which to evaluate the bias bound.
        Yhat_new: (M,) array of predicted targets for the test points.
        X: (N, D) array of train points.
        y: (N,) array of true target values for the train points.
        C: (D, D) array of gradient covariances.
        sigma0_sq: Signal variance. Defaults to 1.
        sigma_sq: Noise variance. Defaults to 0.
        version: One of "exact" or "approximate". Defaults to "exact".

    Returns:
        (M,) array of the bias bound evaluated on the test points
    """
    L = np.linalg.cholesky(C)
    return bias_bound_scaled(
        Z_new=X_new @ L,
        Yhat_new=Yhat_new,
        Z=X @ L,
        y=y,
        sigma0_sq=sigma0_sq,
        sigma_sq=sigma_sq,
        version=version,
    )


def bias_bound_scaled(
    Z_new: np.ndarray,
    Yhat_new: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    sigma0_sq: float,
    sigma_sq: float,
    version="exact",
):
    """Posterior bias bound when C is the identity matrix

    See bias_bound().
    """
    # d: (N * N_new) array, d_ij = \sqrt{\sum_k (Z[i,k] - Z_new[j,k])^2} / sigma0
    d = cdist(Z, Z_new) / sigma0_sq**0.5
    # y_yhat: (N * N_new) array, y_yhat_ij = abs(y[i] - Yhat_new[j])
    y = y.reshape(-1, 1)
    y_yhat = cdist(y, Yhat_new.reshape(-1, 1))
    y_norm = np.sum(y**2) ** 0.5
    if version == "exact":
        return y_norm / 2 + np.min(
            y_yhat
            + np.abs(y) / 2
            + y_norm * sigma0_sq**0.5 / sigma_sq**0.5 * sin_(d / 2),
            axis=0,
        )
    elif version == "approximate":
        ss = sigma0_sq / (sigma0_sq + sigma_sq)
        y_ss_yhat = cdist(y * ss, Yhat_new.reshape(-1, 1))
        return np.min(
            y_ss_yhat + 2 * y_norm * ss * sin_(d / 2),
            axis=0,
        )
    elif version == "practical":
        gamma = 0.6
        beta = np.sqrt(2 * np.log(2 / (1 - gamma)))
        bn = beta / len(y) ** 0.5
        return bn * y_norm + np.min(
            y_yhat + bn * y_norm * sigma0_sq**0.5 / sigma_sq**0.5 * sin_(d / 2),
            axis=0,
        )


def bias_bound_given_var(
    V: np.ndarray, y_hat: np.ndarray, y_ssq_tr: float, sigma_sq: float
):
    """Bias bound, using the exact variance or a variance bound

    Args:
        V: Exact posterior variance at test points.
        y_hat: Regressor predictions at test points.
        y_ssq_tr: Sum of squares of target values for train data.
        sigma_sq: Noise variance.

    Returns:
        Bound for bias |y_hat - y_bar| where y_bar is the posterior mean.
    """
    return (
        (np.abs(y_hat) + np.sqrt(y_ssq_tr + y_hat**2)) * (V + sigma_sq) / (2 * sigma_sq)
    )


def sin_(t):
    """Truncated quarter-sine function"""
    return np.where(t <= np.pi / 2, np.sin(t), 1.0)

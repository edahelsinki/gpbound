import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    RationalQuadratic,
    Matern,
    Kernel,
)
import lzma
from typing import Dict, List
import sys
from pathlib import Path
import pickle
from sklearn.svm import SVR
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).parent.parent))
from gpbound.bound import GPBound, bias_bound_given_var


def run_bound(
    N_tr,
    N_te,
    D,
    C=1,
    sigma0_sq=1,
    sigma_sq=0,
    C_hat=None,
    sigma0_sq_hat=None,
    sigma_sq_hat=None,
    kernel="rbf",
    estimator="gp_rbf",
    sd_tr=None,
    range_te=(0, np.pi / 2),
    n_rep_post=100,
    use_true_params=True,
):
    """Run experiment with RBF kernel."""
    isotropic = is_isotropic(C)
    C_hat = C if C_hat is None else C_hat
    sigma0_sq_hat = sigma0_sq if sigma0_sq_hat is None else sigma0_sq_hat
    sigma_sq_hat = sigma_sq if sigma_sq_hat is None else sigma_sq_hat

    sd_tr = 1 / D**0.5 if sd_tr is None else sd_tr
    X, i_tr = sample_x_normal_radial(
        N_tr=N_tr, N_te=N_te, D=D, sd_tr=sd_tr, range_te=range_te
    )

    ls = (sigma0_sq / C) ** 0.5
    ls_hat = (sigma0_sq_hat / C_hat) ** 0.5
    const = ConstantKernel(sigma0_sq, "fixed")
    match kernel:
        case "rbf":
            kernel_true = const * RBF(ls, "fixed")
        case "rq":
            alpha = 1.0
            kernel_true = const * RationalQuadratic(ls, alpha, "fixed", "fixed")
        case "matern":
            nu = 2.5
            # For nu=2.5, C_matern = sigma0_sq/ls**2 * 5/3.
            ls = ls * (5 / 3) ** 0.5
            kernel_true = const * Matern(ls, "fixed", nu)
        case "cos":
            kernel_true = const * CosKernel(p=1.0)

    f = GPR(kernel_true).sample_y(X, random_state=None).ravel()
    y = f + sigma_sq**0.5 * np.random.standard_normal(f.shape)

    if estimator == "gp_rbf":
        k_hat = ConstantKernel(sigma0_sq_hat, "fixed") * RBF(ls_hat, "fixed")
        f_hat = GPR(k_hat, alpha=sigma_sq_hat).fit(X[i_tr], y[i_tr])
    elif estimator == "rf":
        f_hat = RandomForestRegressor(n_estimators=100).fit(X[i_tr], y[i_tr])
    elif estimator == "svm":
        f_hat = SVR(kernel="rbf", C=1.0, epsilon=0.1).fit(X[i_tr], y[i_tr])
    y_hat = f_hat.predict(X)

    if use_true_params:
        gpb = GPBound(C=C * np.eye(D), sigma0_sq=sigma0_sq, sigma_sq=sigma_sq)
    else:
        gpb = GPBound(
            C=C_hat * np.eye(D), sigma0_sq=sigma0_sq_hat, sigma_sq=sigma_sq_hat
        )
    gpb.fit(X[i_tr], y[i_tr])
    V_hat = gpb.var(X)
    B_hat = gpb.bias(X, y_hat, version="exact")
    B_hat_approx = gpb.bias(X, y_hat, version="approximate")
    B_hat_practical = gpb.bias(X, y_hat, version="practical")

    # Posterior loss, posterior variance, bias
    F_post = GPR(kernel_true, alpha=sigma_sq_hat).fit(X[i_tr], y[i_tr])
    L, V, bias = get_posterior_error(F_post, X, y_hat, sigma_sq, n_rep_post)

    # Posterior loss, posterior variance, bias using fewer train points
    n2 = N_tr // 2
    i_tr_n2 = np.random.choice(i_tr, n2)
    F_post_n2 = GPR(kernel_true, alpha=sigma_sq_hat).fit(X[i_tr_n2], y[i_tr_n2])
    _, V_n2, _ = get_posterior_error(F_post_n2, X, y_hat, sigma_sq, n_rep_post)

    # Bias bound using exact variance
    B_hat_V = bias_bound_given_var(
        V=V, y_hat=y_hat, y_ssq_tr=np.sum(y[i_tr] ** 2), sigma_sq=sigma_sq
    )
    # Bias bound using variance bound
    B_hat_Vhat = bias_bound_given_var(
        V=V_hat, y_hat=y_hat, y_ssq_tr=np.sum(y[i_tr] ** 2), sigma_sq=sigma_sq
    )
    return {
        "V_hat": V_hat.astype(np.float32),
        "V": V.astype(np.float32),
        "B_hat": B_hat.astype(np.float32),
        "B_hat_approx": B_hat_approx.astype(np.float32),
        "B_hat_practical": B_hat_practical.astype(np.float32),
        "bias": bias.astype(np.float32),
        "L": L.astype(np.float32),
        "is_tr": np.repeat([True, False], [N_tr, N_te]),
        "C": C if isotropic else str(C),  # Can't save to feather if C is a matrix.
        "C_hat": C_hat if isotropic else str(C),
        "length_scale": ls if isotropic else str(ls),
        "length_scale_hat": ls_hat if isotropic else str(ls_hat),
        "sigma0_sq": sigma0_sq,
        "sigma0_sq_hat": sigma0_sq_hat,
        "sigma_sq": sigma_sq,
        "sigma_sq_hat": sigma_sq_hat,
        "N_tr": N_tr,
        "N_te": N_te,
        "D": D,
        "kernel": kernel,
        "estimator": estimator,
        "use_true_params": use_true_params,
        "V_n2": V_n2.astype(np.float32),
        "B_hat_V": B_hat_V.astype(np.float32),
        "B_hat_Vhat": B_hat_Vhat.astype(np.float32),
    }


def sample_x_normal_radial(
    N_tr: int, N_te: int, D: int, sd_tr: float = 1, range_te: tuple = (0, 1)
):
    """Sample train data from a spherical Gaussian and test data on a line

    Args:
        N_tr: Number of points in train.
        N_te: Number of points in test.
        D: Number of dimensions.
        sd_tr: Standard deviation of Gaussian for train data.
        range_te: Range of test.

    Returns:
        X: (N_tr+N_te, D) array of train and test data
        i_tr: train indices in X
    """
    X_tr = sd_tr * np.random.standard_normal((N_tr, D))
    X_te = np.column_stack(
        [np.linspace(range_te[0], range_te[1], N_te), np.zeros((N_te, D - 1))]
    )
    X = np.row_stack([X_tr, X_te])
    i_tr = range(N_tr)
    return X, i_tr


def get_posterior_error(
    F_post: GPR, X: np.ndarray, y_hat: np.ndarray, sigma_sq: float, n_rep_post: int
):
    """Posterior error

    Args:
        F_post: Gaussian process posterior as sklearn estimator with predict() and sample_y() methods.
        X: Data covariates as (N, D) array.
        y_hat: Predictions of regression model as (N,) array.
        sigma_sq: Irreducible loss (or prior noise variance).
        n_rep_post: Number of posterior samples to average over for the empirical posterior loss.

    Returns:
        L: (N,) array of empirical expected posterior losses, computed using n_rep_post samples f' from the posterior:
        $L(x) = E(f'(x) - \hat f(x))^2 + \sigma^2$ where f' ~ F' (posterior).
        V: (N,) array of analytical posterior variance of F_post:
        $V(x) = E(f'(x) - \bar f'(x))^2$ where f' ~ F' (posterior) and $\bar f' = E[f']$ (posterior mean).
        bias: (N,) array of the bias: $\bar f'(x) - \hat f(x)$
    """
    y_hat = y_hat if len(y_hat.shape) > 1 else y_hat[:, None]
    try:
        # Posterior samples as (N, n_rep_post) array.
        f_post_samples = F_post.sample_y(X, n_samples=n_rep_post, random_state=None)
    except np.linalg.LinAlgError:
        # Hack: use lower precision X to get around rare cases of non-convergence in np.linalg.svd().
        X = X.astype(np.float32)
        f_post_samples = F_post.sample_y(X, n_samples=n_rep_post, random_state=None)
    # Empirical expected posterior loss
    L = sigma_sq + ((f_post_samples - y_hat) ** 2).mean(axis=1)
    # Empirical posterior variance (unused, shown to contrast L)
    # v = ((f_post_samples - f_post_mean[:, None]) ** 2).mean(axis=1)
    # Analytical posterior mean and stdev
    f_post_mean, sd_post = F_post.predict(X, return_std=True)
    bias = f_post_mean - y_hat.ravel()
    return L, sd_post**2, bias


class CosKernel(Kernel):
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y).
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool, default=False
            Ignored.
        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        """
        X = np.atleast_2d(X)
        Y = X if Y is None else Y
        K = np.cos(self.p * cdist(X, Y))

        if eval_gradient:
            raise NotImplementedError()
        return K

    def diag(self, X):
        return np.diag(self(X))

    def is_stationary(self):
        return True

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


def is_isotropic(C):
    def is_diagonal(X):
        i, j = np.nonzero(X)
        return np.all(i == j)

    if isinstance(C, float) or isinstance(C, int):
        return True
    if isinstance(C, np.ndarray):
        if len(C.shape) == 1 and all(np.diff(C) == 0):
            return True
        if len(C.shape) > 1 and is_diagonal(C) and all(np.diff(np.diag(C)) == 0):
            return True
    return False


def aggregate_dir_of_results(DIR_RES):
    """Read and summarize results in DIR_RES into a data frame."""
    results = []
    for file in os.listdir(DIR_RES):
        if file.endswith(".xz"):
            raw = read_result(Path.joinpath(DIR_RES, file))
            name = os.path.splitext(file)[0]
            df = aggregate_results(raw)
            df.insert(0, "name", name)
            results.append(df)
    return pd.concat(results).sort_values("name").reset_index(drop=True)


def aggregate_results(results: List[Dict]) -> pd.DataFrame:
    """Aggregate results.

    Args:
        results (list): list of dicts. Each dict is a run of run_bound().

    Returns:
        pd.DataFrame: rows are metrics (bound - value), columns are aggregates
        (mean, 5% quantile, 95% quantile) over all repetitions and over points
        in each repetition.
    """
    df = pd.concat([pd.DataFrame(r) for r in results])
    df = df.groupby("rep", group_keys=False).apply(lambda x: x[~x.is_tr])

    def median(x):
        return np.nanmedian(x, axis=0)

    def q5(x):
        if all(np.isinf(x)):
            # Avoid warning from np.quantile.
            return np.inf
        return np.nanquantile(x, q=0.05, axis=0)

    def q95(x):
        if all(np.isinf(x)):
            # Avoid warning from np.quantile.
            return np.inf
        return np.nanquantile(x, q=0.95, axis=0)

    B = np.abs(df["bias"])
    V = df["V"]
    V_hat = df["V_hat"]
    B_hat_exact = df["B_hat"]
    B_hat_approx = df["B_hat_approx"]
    B_hat_practical = df["B_hat_practical"]
    B_hat_Vhat = df["B_hat_Vhat"]
    V_n2 = df["V_n2"]
    sigma0_sq = df["sigma0_sq"]
    df_agg = (
        pd.DataFrame(
            {
                "V_hat - V": V_hat - V,
                "V_n2 - V": V_n2 - V,
                "sigma0_sq - V": sigma0_sq - V,
                "B_hat - B_abs": B_hat_exact - B,
                "B_hat_Vhat - B_abs": B_hat_Vhat - B,
                "B_hat_approx - B_abs": B_hat_approx - B,
                "B_hat_practical - B_abs": B_hat_practical - B,
                "V": V,
                "B_abs": B,
            }
        )
        .agg([median, q5, q95, np.std])
        .reset_index(names="stat")
        .assign(id="1")
        .pivot(index="id", columns="stat")
        .reset_index(drop=True)
    )
    df_agg.columns = [
        col[0] if col[1] == "" else " ".join(col)
        for col in df_agg.columns.to_flat_index()
    ]
    df_agg["n_rep"] = len(df.rep.unique())
    return df_agg


def save_result(result, filename: Path):
    with lzma.open(filename, "wb") as f:
        pickle.dump(result, f)


def read_result(filename: Path):
    with lzma.open(filename, "rb") as f:
        raw = pickle.load(f)
    return raw


def make_grid_names(grid, name):
    return [f"{name}{i:03d}" for i, _ in enumerate(grid)]


def make_df_params(grids):
    df = pd.DataFrame(
        {
            name: params
            for method, grid in grids.items()
            for params, name in zip(grid, make_grid_names(grid, method))
        }
    ).T.reset_index(names="name")
    return df


grid_gp_rbf = [
    {
        "N_tr": 50,
        "N_te": 100,
        "D": D,
        "C": 1.0,
        "C_hat": C_hat,
        "sigma_sq": sigma_sq,
        "sd_tr": 1 / D**0.5,
        "kernel": kernel,
        "estimator": "gp_rbf",
        "use_true_params": False,
    }
    for C_hat in [1, 4]
    for D in [1, 2, 10, 20]
    for sigma_sq in [1, 0.1]
    for kernel in ["rbf", "rq", "matern"]
]

grid_rf = [
    {
        "N_tr": 50,
        "N_te": 100,
        "D": D,
        "C": 1.0,
        "C_hat": C_hat,
        "sd_tr": 1 / D**0.5,
        "sigma_sq": sigma_sq,
        "kernel": kernel,
        "estimator": "rf",
        "use_true_params": False,
    }
    for C_hat in [1, 4]
    for D in [1, 2, 10, 20]
    for sigma_sq in [1, 0.1]
    for kernel in ["rbf", "rq", "matern"]
]

grid_svm = [
    {
        "N_tr": 50,
        "N_te": 100,
        "D": D,
        "C": 1.0,
        "C_hat": C_hat,
        "sd_tr": 1 / D**0.5,
        "sigma_sq": sigma_sq,
        "kernel": kernel,
        "estimator": "svm",
        "use_true_params": False,
    }
    for C_hat in [1, 4]
    for D in [1, 2, 10, 20]
    for sigma_sq in [1, 0.1]
    for kernel in ["rbf", "rq", "matern"]
]

run_exp_fns = {
    "gp_rbf": run_bound,
    "rf": run_bound,
    "svm": run_bound,
}

grids = {
    "gp_rbf": grid_gp_rbf,
    "rf": grid_rf,
    "svm": grid_svm,
}

if __name__ == "__main__":
    import os
    from datetime import datetime
    from utils import load_config

    config = load_config(Path(__file__).parent.parent / "config.json")
    DIR_RES = config["dir_results"] / "bound"

    args = sys.argv
    idx = args[1] if args and len(args) > 1 else -1
    method = args[2] if args and len(args) > 2 else "gp_rbf"
    n_rep = int(args[3]) if args and len(args) > 3 else 200
    debug = n_rep < 1
    summarize = idx == -1

    if debug:
        n_rep = 2
        DIR_RES = config["dir_results"] / "tmp"

    run_exp = run_exp_fns[method]
    grid = grids[method]
    grid_names = make_grid_names(grid, method)

    t_start = datetime.now()
    print(f"Bound experiment. {t_start.strftime('%Y-%m-%d %T')}")
    print(f"Results will be saved to {DIR_RES}.")
    DIR_RES.mkdir(exist_ok=True, parents=True)
    if not summarize:
        idxs = [int(idx)] if idx != "all" else range(len(grid))
        for i in idxs:
            name = grid_names[i]
            params = grid[i]
            file_out = Path.joinpath(DIR_RES, f"{name}.xz")
            print(f"{name}: {params}. n_rep={n_rep}.")
            np.random.seed(2023 + i**2 * 2023 + len(method))
            results = []
            t_last_result = datetime.fromordinal(1)
            for rep in range(1, n_rep + 1):
                res = run_exp(**params) | {"rep": rep}
                results.append(res)
                t_delta = (datetime.now() - t_last_result).total_seconds()
                if t_delta > 60 or rep == n_rep:
                    save_result(results, file_out)
                    t_last_result = datetime.now()
    else:
        print(f"Summarizing results.")
        file_out = Path.joinpath(DIR_RES, "tbl_bound.feather")
        df_results = aggregate_dir_of_results(DIR_RES)
        df_params = make_df_params(grids)
        (
            df_results.merge(df_params)
            .sort_values("name")
            .reset_index(drop=True)
            .to_feather(file_out)
        )
        print(f"Saved to {file_out}.")

    t_end = datetime.now()
    print(
        f"Done. ({t_end.strftime('%Y-%m-%d %T')}, {(t_end - t_start).total_seconds(): .0f} sec)"
    )

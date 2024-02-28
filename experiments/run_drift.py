"""Run drift experiment with real data"""

import os
import pickle
from timeit import default_timer
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from pathlib import Path
import sys
import re
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).parent.parent))
from gpbound.bound import GPBound
from gpbound.gradient import estimate_c
from data.data import get_dataset


def run_drift(
    dataset: str,
    regressor: str,
    pct_tr: float,
    pct_va: float = None,
    h: float = None,
    split_method: str = "impurity",
    DIR_DATA=None,
):
    X, y, X_names = get_dataset(dataset, DIR_DATA, return_names=True, scale_=True)
    pct_va = pct_va if pct_va is not None else pct_tr
    N, D = X.shape
    N_tr, N_va = int(N * pct_tr), int(N * pct_va)
    N_te = N - N_tr - N_va
    t0 = default_timer()
    i_tr, i_va, i_te, split_variable = get_split(
        X=X,
        y=y,
        n_tr=N_tr,
        n_va=N_va,
        regressor=regressor,
        method=split_method,
    )
    if split_variable is not None:
        X = np.delete(X, split_variable, axis=1)
    t_split = default_timer() - t0

    t0 = default_timer()
    if regressor == "rf":
        estimator = RandomForestRegressor(random_state=0)
        param_grid = {
            "n_estimators": [100, 500],
            "min_samples_leaf": [1, 5],
            "max_features": [None, "sqrt"],
            "max_samples": [0.5, 1.0],
        }
    elif regressor == "svm":
        estimator = SVR()
        # For epsilon > epsilon_max, all train data are inside the epsilon-tube.
        epsilon_max = (np.max(y[i_tr]) - np.min(y[i_tr])) / 2
        param_grid = {
            "C": np.logspace(-5, 5, 10, base=np.e),
            "epsilon": np.logspace(-3, np.min([1, epsilon_max]), 10, base=np.e),
        }
    search = GridSearchCV(estimator, param_grid, scoring="neg_mean_squared_error", cv=5)
    search.fit(X[i_tr], y[i_tr])
    f_hat = search.best_estimator_.predict
    f_hat_params = search.best_params_
    y_hat = f_hat(X)
    t_fhat = default_timer() - t0

    t0 = default_timer()
    h = 1e-6 if h is None else h
    C_hat = estimate_c(X=X[i_tr], f=f_hat, h=h)
    C_hat = C_hat + 1e-10 * np.eye(C_hat.shape[0])  # Ensure positive definiteness.
    t_c = default_timer() - t0

    t0 = default_timer()
    sigma0_sq_hat = 1.0  # Assume equal to 1 because y is normalized to unit variance.
    # MSE of regressor selected with grid search CV.
    sigma_sq_hat = np.abs(search.best_score_)
    gpb = GPBound(C=C_hat, sigma0_sq=sigma0_sq_hat, sigma_sq=sigma_sq_hat)
    gpb.fit(X[i_tr], y[i_tr])
    V_hat = gpb.var(X)
    B_hat = gpb.bias(X, y_hat)
    B_hat_approx = gpb.bias(X, y_hat, version="approximate")
    B_hat_pract = gpb.bias(X, y_hat, version="practical")
    t_bound = default_timer() - t0

    # Baseline: Euclidean distance from training centroid.
    dist = np.sqrt(((X - X[i_tr].mean(axis=0)) ** 2).sum(axis=1))
    # Baseline 2: Euclidean distance from k:th closest training point.
    k = 5
    dist_k = np.sort(cdist(X, X[i_tr]), axis=1)[:, k]

    return {
        "V_hat": V_hat.astype(np.float32),
        "B_hat": B_hat.astype(np.float32),
        "B_hat_approx": B_hat_approx.astype(np.float32),
        "B_hat_practical": B_hat_pract.astype(np.float32),
        "sq_error": ((y - y_hat) ** 2).astype(np.float32),
        "y_hat": y_hat.astype(np.float32),
        "dist": dist.astype(np.float32),
        "dist_k": dist_k.astype(np.float32),
        "C_hat": np.diagonal(C_hat),
        "is_tr": np.array([i in i_tr for i in range(X.shape[0])]),
        "is_va": np.array([i in i_va for i in range(X.shape[0])]),
        "t_split": t_split,
        "t_fhat": t_fhat,
        "t_c": t_c,
        "t_bound": t_bound,
        "f_hat_params": f_hat_params,
        "sigma_sq_hat": sigma_sq_hat,
        "sigma0_sq_hat": sigma0_sq_hat,
        "split_variable": split_variable,
        "split_method": split_method,
        "X_names": X_names,
        "dataset": dataset,
        "regressor": regressor,
        "N_tr": N_tr,
        "N_te": N_te,
        "D": D,
        "h": h,
    }


def get_split(X, y, n_tr, n_va, regressor, method="pfi"):
    match method:
        case "pfi":
            return split_by_rf_feat_importance(
                X=X, y=y, n_tr=n_tr, n_va=n_va, metric="pfi"
            )
        case "impurity":
            return split_by_rf_feat_importance(
                X=X, y=y, n_tr=n_tr, n_va=n_va, metric="impurity"
            )
        case "seq":
            I = range(len(X))
            i_tr, i_va = train_test_split(
                I[: (n_tr + n_va)], test_size=n_va, train_size=n_tr, random_state=None
            )
            i_te = I[(n_tr + n_va) :]
            return i_tr, i_va, i_te, None
        case "loss_diff":
            return split_by_loss_diff(
                X=X, y=y, n_tr=n_tr, n_va=n_va, regressor=regressor, m_repeats=5
            )


def split_by_rf_feat_importance(X, y, n_tr, n_va=None, metric="impurity"):
    n_va = n_va if n_va is not None else n_tr
    rf = RandomForestRegressor(random_state=0).fit(X, y)
    if metric == "impurity":
        j = np.argmax(rf.feature_importances_)
    elif metric == "pfi":
        pfi = permutation_importance(rf, X, y, n_repeats=5, random_state=0)
        j = np.argmax(pfi["importances_mean"])
    I = np.argsort(X[:, j])
    i_tr, i_va = train_test_split(
        I[: (n_tr + n_va)], test_size=n_va, train_size=n_tr, random_state=0
    )
    i_te = I[(n_tr + n_va) :]
    return i_tr, i_va, i_te, j


def split_by_loss_diff(X, y, n_tr, n_va=None, regressor="rf", m_repeats=1):
    def split_and_train(X, y, j, n_tr, n_va, estimator, random_state=None) -> dict:
        """Split to train-validation-test by j:th variable and compute regressor losses."""
        I = np.argsort(X[:, j])
        X_ = np.delete(X, j, axis=1)
        i_trva, i_te = I[: (n_tr + n_va)], I[(n_tr + n_va) :]
        i_tr, i_va = train_test_split(
            i_trva, test_size=n_va, train_size=n_tr, random_state=random_state
        )
        f_hat = estimator.fit(X_[i_tr], y[i_tr])
        y_hat = f_hat.predict(X_)
        sq_error = (y_hat - y) ** 2
        loss_va = sq_error[i_va].mean()
        loss_te = sq_error[i_te].mean()
        return {
            "i_tr": i_tr,
            "i_va": i_va,
            "i_te": i_te,
            "j": j,
            "loss_tr": sq_error[i_tr].mean(),
            "loss_va": loss_va,
            "loss_te": loss_te,
            "drift": loss_te - loss_va,
        }

    if regressor == "rf":
        estimator = RandomForestRegressor(random_state=None)
    elif regressor == "svm":
        estimator = SVR()

    n_va = n_va if n_va is not None else n_tr

    res = []
    for j in range(X.shape[1]):
        for rep in range(m_repeats):
            res.append(
                split_and_train(
                    X=X, y=y, j=j, n_tr=n_tr, n_va=n_va, estimator=estimator
                )
                | {"rep": rep}
            )
    max_drift = max(res, key=lambda x: x["drift"])
    return tuple(max_drift[key] for key in ["i_tr", "i_va", "i_te", "j"])


def save_result(result, filename: Path):
    with open(filename, "wb") as f:
        pickle.dump(result, f)


def read_result(filename: Path):
    with open(filename, "rb") as f:
        raw = pickle.load(f)
    return raw


def summarize_datasets(grid, DIR_DATA):
    df = pd.DataFrame(grid)[["dataset", "pct_tr"]].drop_duplicates()
    l = []
    for i, row in df.iterrows():
        X, y = get_dataset(row.dataset, DIR_DATA)
        l.append(
            {
                "dataset": row.dataset,
                "N": X.shape[0],
                "N_tr": int(X.shape[0] * row.pct_tr),
                "D": X.shape[1],
            }
        )
    datasets = pd.DataFrame(l)
    datasets["dataset"] = datasets["dataset"].str.replace("_", "\_")
    datasets.sort_values(by="dataset", inplace=True)
    return datasets


def summarize_drift(DIR_RES):
    df_list = []
    for file in os.listdir(DIR_RES):
        if re.search("^drift\d+\.pkl$", file):
            res = read_result(Path.joinpath(DIR_RES, file))
            name = os.path.splitext(file)[0]
            df_res = pd.DataFrame([res])
            df_res.insert(0, "name", name)
            df_list.append(df_res)
    df = pd.concat(df_list).sort_values(by=["name"]).reset_index(drop=True)
    return df


def make_grid_names(grid, name):
    return [f"{name}{i:02d}" for i, _ in enumerate(grid)]


def make_df_params(grids):
    df = pd.DataFrame(
        {
            name: params
            for method, grid in grids.items()
            for params, name in zip(grid, make_grid_names(grid, method))
        }
    ).T.reset_index(names="name")
    return df


datasets = [
    "autompg",
    "airfoil",
    "airquality",
    "winequality",
    "yearpredictionmsd",
    "qm9",
    "oe62",
    "aa",
    "abalone",
    "california",
    "concrete",
    "cpu_small",
]

grid = [
    {
        "dataset": dataset,
        "regressor": regressor,
        "pct_tr": 0.3,
        "pct_va": 0.3,
        "h": 0.01 if regressor == "rf" else None,
    }
    for dataset in datasets
    for regressor in ["rf", "svm"]
]
grid_names = make_grid_names(grid, "drift")


if __name__ == "__main__":
    from datetime import datetime
    from utils import load_config

    config = load_config(Path(__file__).parent.parent / "config.json")
    DIR_DATA = config["dir_data"]
    DIR_RES = config["dir_results"] / "drift"

    args = sys.argv
    idx = args[1] if args and len(args) > 1 else "-1"
    opt = args[2] if args and len(args) > 2 else "drift"
    debug = opt == "debug"
    summarize = idx == "-1"

    if debug:
        DIR_RES = config["dir_results"] / "tmp"

    t_start = datetime.now()
    print(f"Drift experiment. {t_start.strftime('%Y-%m-%d %T')}")
    print(f"Results to {DIR_RES}.")
    DIR_RES.mkdir(exist_ok=True, parents=True)

    if not summarize:
        idxs = [int(idx)] if idx != "all" else range(len(grid))
        for i in idxs:
            params = grid[i]
            name = grid_names[i]
            file_out = Path.joinpath(DIR_RES, f"{name}.pkl")
            print(f"{name}: {params}")
            np.random.seed((i + 2023) * 2023)
            res = run_drift(**params, DIR_DATA=DIR_DATA)
            save_result(res, file_out)
    else:
        if opt == "datasets":
            DIR_TBL = config["dir_tables"]
            print(f"Summarizing datasets to {DIR_TBL}.")
            df_data = summarize_datasets(grid, DIR_DATA)
            (
                df_data.rename(
                    columns={
                        "dataset": "Dataset",
                        "N": "$N$",
                        "N_tr": "$N_{tr}$",
                        "D": "$p$",
                    }
                )
                .style.hide(axis="index")
                .set_properties(
                    subset=["Dataset"],
                    **{"textsc": "--rwrap"},
                )
                .to_latex(DIR_TBL / "datasets.tex", hrules=True)
            )
        elif opt == "drift":
            print(f"Summarizing results.")
            drift = summarize_drift(DIR_RES)
            drift.to_feather(Path.joinpath(DIR_RES, "tbl_drift.feather"))

    t_end = datetime.now()
    t_delta = t_end - t_start
    print(
        f"Done. ({t_end.strftime('%Y-%m-%d %T')}, {t_delta.total_seconds(): .0f} sec)"
    )

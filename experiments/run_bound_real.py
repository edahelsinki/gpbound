"""Run bound validity experiment with real data"""

import os
import pickle
from timeit import default_timer
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from pathlib import Path
import sys
import re

sys.path.insert(0, str(Path(__file__).parent.parent))
from gpbound.bound import GPBound
from gpbound.gradient import estimate_c
from data.data import get_dataset


def run_bound_real(
    dataset: str,
    regressor: str,
    pct_tr: float,
    h: float = None,
    DIR_DATA=None,
):
    X, y = get_dataset(dataset, DIR_DATA)

    N, D = X.shape
    N_tr, N_te = int(N * pct_tr), N - int(N * pct_tr)
    t0 = default_timer()
    i_tr, i_te = train_test_split(np.arange(N), train_size=N_tr, test_size=N_te)
    t_split = default_timer() - t0

    t0 = default_timer()
    if regressor == "rf":
        estimator = RandomForestRegressor(random_state=0)
    elif regressor == "svm":
        estimator = SVR()
    estimator.fit(X[i_tr], y[i_tr])
    f_hat = estimator.predict
    y_hat = f_hat(X)
    sq_error = (y - y_hat) ** 2
    t_fhat = default_timer() - t0

    t0 = default_timer()
    h = 1e-6 if h is None else h
    C_hat = estimate_c(X=X[i_tr], f=f_hat, h=h)
    C_hat = C_hat + 1e-10 * np.eye(C_hat.shape[0])  # Ensure positive definiteness.
    t_c = default_timer() - t0

    t0 = default_timer()
    sigma0_sq_hat = 1.0  # Assume equal to 1 because y is normalized to unit variance.
    sigma_sq_hat = sq_error[i_te].mean()  # MSE of regressor
    gpb = GPBound(C=C_hat, sigma0_sq=sigma0_sq_hat, sigma_sq=sigma_sq_hat)
    gpb.fit(X[i_tr], y[i_tr])
    V_hat = gpb.var(X)
    B_hat = gpb.bias(X, y_hat, version="exact")
    B_hat_approx = gpb.bias(X, y_hat, version="approximate")
    B_hat_pract = gpb.bias(X, y_hat, version="practical")
    t_bound = default_timer() - t0

    return {
        "V_hat": V_hat.astype(np.float32),
        "B_hat": B_hat.astype(np.float32),
        "B_hat_approx": B_hat_approx.astype(np.float32),
        "B_hat_practical": B_hat_pract.astype(np.float32),
        "sq_error": sq_error.astype(np.float32),
        "C_hat": np.diagonal(C_hat),
        "is_tr": np.array([i in i_tr for i in range(X.shape[0])]),
        "t_split": t_split,
        "t_fhat": t_fhat,
        "t_c": t_c,
        "t_bound": t_bound,
        "sigma_sq_hat": sigma_sq_hat,
        "sigma0_sq_hat": sigma0_sq_hat,
        "dataset": dataset,
        "regressor": regressor,
        "N_tr": N_tr,
        "N_te": N_te,
        "D": D,
        "h": h,
    }


def save_result(result, filename: Path):
    with open(filename, "wb") as f:
        pickle.dump(result, f)


def read_result(filename: Path):
    with open(filename, "rb") as f:
        raw = pickle.load(f)
    return raw


def aggregate_results(results: List[Dict]) -> pd.DataFrame:
    """Aggregate results.

    Args:
        results (list): list of dicts. Each dict is a run of run_bound_real().

    Returns:
        pd.DataFrame: rows are metrics (bound - value), columns are aggregates
        (mean, 5% quantile, 95% quantile) over all repetitions and over points
        in each repetition.
    """

    def median(x):
        return np.median(x, axis=0)

    def q_low(x):
        return np.quantile(x, q=0.01, axis=0)

    def q_high(x):
        return np.quantile(x, q=0.99, axis=0)

    df = pd.DataFrame(results)
    df = df.explode(
        [
            "V_hat",
            "B_hat",
            "B_hat_practical",
            "B_hat_approx",
            "sq_error",
            "is_tr",
        ]
    )
    df = df[~df.is_tr]

    e = df["sq_error"]
    bound_exact = df["V_hat"] + df["B_hat"] ** 2
    bound_approx = df["V_hat"] + df["B_hat_approx"] ** 2
    bound_pract = df["V_hat"] + df["B_hat_practical"] ** 2
    df_agg = (
        pd.DataFrame(
            {
                "sq_error": e,
                "bound_exact - sq_error": bound_exact - e,
                "bound_approx - sq_error": bound_approx - e,
                "bound_pract - sq_error": bound_pract - e,
                "(bound_exact - sq_error) / sq_error": (bound_exact - e) / e,
                "(bound_approx - sq_error) / sq_error": (bound_approx - e) / e,
                "(bound_pract - sq_error) / sq_error": (bound_pract - e) / e,
            }
        )
        .agg([median, q_low, q_high])
        .reset_index(names="stat")
        .assign(id="1")
        .pivot(index="id", columns="stat")
        .reset_index(drop=True)
    )
    df_agg.columns = [
        col[0] if col[1] == "" else " ".join(col)
        for col in df_agg.columns.to_flat_index()
    ]
    return df_agg


def summarize_bound_real(DIR_RES):
    df_list = []
    for file in os.listdir(DIR_RES):
        if re.search("^bound_real\d+\.pkl$", file):
            res = read_result(Path.joinpath(DIR_RES, file))
            name = os.path.splitext(file)[0]
            df_res = aggregate_results(res)
            df_res.insert(0, "name", name)
            df_list.append(df_res)
    df = pd.concat(df_list).sort_values(by=["name"]).reset_index(drop=True)
    return df


def make_grid_names(grid, name="bound_real"):
    return [f"{name}{i:02d}" for i, _ in enumerate(grid)]


def make_df_params(grid, name="bound_real"):
    df = pd.DataFrame(
        {name: params for params, name in zip(grid, make_grid_names(grid, name))}
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
        "h": 0.01 if regressor == "rf" else None,
    }
    for dataset in datasets
    for regressor in ["rf", "svm"]
]

grid_names = make_grid_names(grid, "bound_real")

if __name__ == "__main__":
    from datetime import datetime
    from utils import load_config

    config = load_config(Path(__file__).parent.parent / "config.json")
    DIR_DATA = config["dir_data"]
    DIR_RES = config["dir_results"] / "bound_real"

    args = sys.argv
    idx = args[1] if args and len(args) > 1 else "-1"
    n_rep = int(args[2]) if args and len(args) > 2 else 1
    debug = n_rep < 0
    summarize = idx == "-1"

    if debug:
        DIR_RES = config["dir_results"] / "tmp"

    t_start = datetime.now()
    print(f"Real data experiment. {t_start.strftime('%Y-%m-%d %T')}")
    print(f"Results to {DIR_RES}.")
    DIR_RES.mkdir(exist_ok=True, parents=True)

    if not summarize:
        idxs = [int(idx)] if idx != "all" else range(len(grid))
        for i in idxs:
            params = grid[i]
            name = grid_names[i]
            file_out = Path.joinpath(DIR_RES, f"{name}.pkl")
            print(f"{name}: {params}. n_rep={n_rep}")
            np.random.seed((i + 2023) * 2023)
            results = []
            t_last_result = datetime.fromordinal(1)
            for rep in range(1, n_rep + 1):
                res = run_bound_real(**params, DIR_DATA=DIR_DATA) | {"rep": rep}
                results.append(res)
                t_delta = (datetime.now() - t_last_result).total_seconds()
                if t_delta > 60 or rep == n_rep:
                    save_result(results, file_out)
                    t_last_result = datetime.now()
    else:
        print(f"Summarizing results.")
        DIR_TBL = config["dir_tables"]
        file_out = Path.joinpath(DIR_RES, "tbl_bound_real.feather")
        bound_real = summarize_bound_real(DIR_RES)
        (
            bound_real.merge(make_df_params(grid))
            .sort_values("name")
            .reset_index(drop=True)
            .to_feather(file_out)
        )

    t_end = datetime.now()
    t_delta = t_end - t_start
    print(
        f"Done. ({t_end.strftime('%Y-%m-%d %T')}, {t_delta.total_seconds(): .0f} sec)"
    )

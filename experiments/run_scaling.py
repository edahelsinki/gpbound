import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern
import sys
from pathlib import Path
from timeit import default_timer
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from gpbound.bound import GPBound


def run_gp(N, D, job_index=0, method="rbf"):
    seed = 42 + job_index + N * D + D
    np.random.seed(seed)
    X = np.random.uniform(0, 1, (N, D))
    y = X[:, 0]
    match method:
        case "rbf":
            k = RBF(1.0, "fixed")
        case "rq":
            k = RationalQuadratic(1.0, 1.0, "fixed", "fixed")
        case "matern":
            k = Matern(1.0, "fixed", nu=2.5)
    time = default_timer()
    gp = GPR(kernel=k).fit(X, y)
    time_fit = default_timer() - time
    _ = gp.predict(X, return_std=True)
    time = default_timer() - time
    return {
        "time": time,
        "time_fit": time_fit,
        "N": N,
        "D": D,
        "method": method,
    }


def run_bound(N, D, job_index=0):
    seed = 42 + job_index + N * D + D
    np.random.seed(seed)
    X = np.random.uniform(0, 1, (N, D))
    time = default_timer()
    gpb = GPBound(C=np.eye(D), sigma0_sq=1.0, sigma_sq=0.0).fit(X)
    time_fit = default_timer() - time
    _ = gpb.var(X)
    time = default_timer() - time
    return {"time": time, "time_fit": time_fit, "N": N, "D": D, "method": "bound"}


run_exp_fns = {
    "bound": run_bound,
    "rbf": lambda N, D, job_index: run_gp(N, D, job_index, method="rbf"),
    "rq": lambda N, D, job_index: run_gp(N, D, job_index, method="rq"),
    "matern": lambda N, D, job_index: run_gp(N, D, job_index, method="matern"),
}

if __name__ == "__main__":
    from utils import load_config

    config = load_config(Path(__file__).parent.parent / "config.json")
    DIR_RES = config["dir_results"] / "scaling"

    job_index = int(sys.argv[1]) if len(sys.argv) > 1 else "-1"
    method = sys.argv[2] if len(sys.argv) > 2 else "bound"
    summarize = job_index == "-1"

    t_start = datetime.now()
    print(f"Scaling experiment ({t_start.strftime('%Y-%m-%d %T')})")
    print(f"Results will be saved to {DIR_RES}.")
    DIR_RES.mkdir(exist_ok=True, parents=True)

    if not summarize:
        filename = DIR_RES / Path(f"{method}_{job_index:02d}.pkl")
        sizes = [
            (N, D)
            for N in [100, 500, 1000, 2500, 5000, 7500, 10000]
            for D in [1, 2, 5, 15, 25]
        ]
        run_exp = run_exp_fns[method]

        print(f"job={job_index}, method={method}")

        results = []
        t_last_result = datetime.fromordinal(1)
        for i, (N, D) in enumerate(sizes, start=1):
            print(f"{i}/{len(sizes)}. N={N}. D={D}.")
            res = run_exp(N, D, job_index)
            results.append(res)
            t_delta = (datetime.now() - t_last_result).total_seconds()
            if t_delta > 60 or i == len(sizes):
                df = pd.DataFrame(results)
                df.to_pickle(path=filename)
    else:
        import os

        def q5(x):
            return np.quantile(x, q=0.05, axis=0)

        def q95(x):
            return np.quantile(x, q=0.95, axis=0)

        print("Summarizing results.")
        file_table = DIR_RES / "tbl_scaling.feather"
        results_all = []
        for filename in os.listdir(DIR_RES):
            if filename.endswith(".pkl"):
                with open(Path.joinpath(DIR_RES, filename), "rb") as f:
                    results_all.append(pd.read_pickle(f))
        df_all = pd.concat(results_all, axis=0)
        df_agg = (
            df_all.groupby(by=["N", "D", "method"])
            .aggregate([np.mean, q5, q95, np.std])
            .reset_index()
        )
        df_agg.columns = [C[0] if C[1] == "" else "_".join(C) for C in df_agg.columns]
        df_agg.to_feather(file_table)
        print(f"Saved to {file_table}.")

    t_end = datetime.now()
    t_delta = t_end - t_start
    print(
        f"Done. ({t_end.strftime('%Y-%m-%d %T')}, {t_delta.total_seconds(): .0f} sec)"
    )

from pathlib import Path
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).parent.parent))
from gpbound.bound import GPBound
from experiments.run_bound import CosKernel

palette_kernels = {
    "cos": "black",
    "rbf": "#377eb8",
    "rq": "#7fc97f",
    "matern": "#e41a1c",
}
# Use Type 42 (a.k.a. TrueType) fonts (instead of the default Type 3).
plt.rcParams["pdf.fonttype"] = 42


def plot_kernels(file=None, width=5, height=3):
    ls = 1
    zero = np.zeros((1, 1))
    xx = np.linspace(0, np.pi, num=100).reshape(-1, 1)
    plt.rc("axes", labelsize=15)
    plt.rc("xtick", labelsize=15)
    plt.rc("ytick", labelsize=15)
    fig, ax = plt.subplots(figsize=(width, height))
    plt.plot(xx, np.cos(xx), label="cos", c=palette_kernels["cos"])
    plt.plot(xx, RBF(ls)(xx, zero), c=palette_kernels["rbf"], label="rbf")
    plt.plot(
        xx,
        RationalQuadratic(ls, alpha=1)(xx, zero),
        c=palette_kernels["rq"],
        label="rq",
    )
    plt.plot(
        xx,
        Matern(ls * np.sqrt(5 / 3), nu=2.5)(xx, zero),
        c=palette_kernels["matern"],
        label="matern",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend()
    plt.axhline(0, lw=1, c="black", ls="dotted")
    plt.xlabel("$\\Vert p-q \\Vert$")
    plt.ylabel("$k(p, q)$")
    plt.xticks([0, np.pi / 2, np.pi], ["0", "$\pi/2$", "$\pi$"])
    plt.tight_layout()
    if file:
        plt.savefig(file, dpi=500)
    else:
        plt.show()


def plot_kernel_variances_1d(file=None, height=3, width=5):
    np.random.seed(2023)
    results = []
    C = 1
    sigma0_sq = 1
    rC_list = [1, 4]
    rs0_list = [1]
    for kernel in ["rbf", "rq", "matern", "cos"]:
        for rC in rC_list:
            for rs0 in rs0_list:
                res = run_bound_1d(
                    C_hat=C * rC,
                    C=C,
                    sigma0_sq=sigma0_sq,
                    sigma0_sq_hat=sigma0_sq * rs0,
                    kernel=kernel,
                )
                results.append(res | {"rC": rC} | {"rs0": rs0})

    style_r = {
        (rC_list[0], rs0_list[0]): "solid",
        (rC_list[1], rs0_list[0]): "dashed",
        # (rC_list[1], rs0_list[1]): "dotted",
    }
    plt.rc("axes", labelsize=15)
    plt.rc("xtick", labelsize=15)
    plt.rc("ytick", labelsize=15)
    fig, ax = plt.subplots(figsize=(width, height))
    for res in results:
        ax.plot(
            res["V"],
            res["V_hat"],
            c=palette_kernels[res["kernel"]],
            ls=style_r[(res["rC"], res["rs0"])],
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("$\mathcal{V}$")
    plt.ylabel("$\hat{\mathcal{V}}$")
    plt.axline((0, 0), slope=1, ls="dotted", c="black", lw=0.1)
    plt.xticks([0, 0.5, 1])
    plt.yticks([0, 0.5, 1])
    plt.tight_layout()

    # df = pd.DataFrame(results).explode(["V", "V_hat"])
    # df = df.query("~(C != C_hat & sigma0_sq != sigma0_sq_hat)")
    # plt.figure(figsize=(width, height))
    # g = sns.lineplot(
    #     df,
    #     x="V",
    #     y="V_hat",
    #     hue="kernel",
    #     style="rC",
    #     palette=palette_kernels,
    # )
    # plt.xlabel("$\mathcal{V}$")
    # plt.ylabel("$\hat{\mathcal{V}}$")
    # plt.axline((0, 0), slope=1, ls="dotted", c="black", lw=0.1)
    # plt.box(False)
    # plt.xticks([0, 0.5, 1])
    # plt.yticks([0, 0.5, 1])
    # # plt.legend()
    # g.legend_.remove()
    # plt.tight_layout()
    if file:
        plt.savefig(file, dpi=500)
    else:
        plt.show()


def plot_directionality(file=None, width=10, height=3):
    res = run_directionality(
        N_tr=1,
        N_te=50,
        D=2,
        C=np.array([1, 4]),
        sd_tr=1e-10,
        sigma_sq=1e-3,
        dims_te=(0, 1),
        range_te=(0, np.pi / 2),
    )
    df = pd.DataFrame(res)
    i_te1 = df.id == 1
    i_te2 = df.id == 2

    fig, ax = plt.subplots(figsize=(width, height))
    plt.plot(df.d[i_te1], df.V[i_te1], c="blue", label="$\mathcal{V}_1$")
    plt.plot(
        df.d[i_te1],
        df.V_hat[i_te1],
        ls="dashed",
        c="blue",
        label="$\hat{\mathcal{V}}_1$",
    )
    plt.plot(df.d[i_te2], df.V[i_te2], c="orange", label="$\mathcal{V}_2$")
    plt.plot(
        df.d[i_te2],
        df.V_hat[i_te2],
        ls="dashed",
        c="orange",
        label="$\hat{\mathcal{V}}_2$",
    )
    plt.legend()
    plt.xlabel("Distance $|p-q|$")
    plt.ylabel("Variance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if file:
        plt.savefig(file, dpi=500)
    else:
        plt.show()


def run_bound_1d(
    C_hat=None,
    sigma0_sq_hat=None,
    C=1,
    sigma0_sq=1,
    sigma_sq=0,
    kernel="rbf",
    N_te=100,
    range_te=(0, np.pi / 2),
):
    C_hat = C if C_hat is None else C_hat
    sigma0_sq_hat = sigma0_sq if sigma0_sq_hat is None else sigma0_sq_hat
    ls = (sigma0_sq / C) ** 0.5
    match kernel:
        case "rbf":
            k_true = RBF(ls, "fixed")
        case "rq":
            alpha = 1.0
            k_true = RationalQuadratic(ls, alpha, "fixed", "fixed")
        case "matern":
            nu = 2.5
            ls = ls * np.sqrt(5 / 3)  # Because C_matern = sigma0_sq/ls**2 * 5/3
            k_true = Matern(ls, "fixed", nu)
        case "cos":
            k_true = CosKernel(p=1.0)
    X_tr = np.zeros((1, 1))
    X_te = np.linspace(range_te[0], range_te[1], N_te).reshape(-1, 1)
    X = np.vstack([X_tr, X_te])
    i_tr = range(1)
    f = GPR(k_true).sample_y(X, random_state=None).ravel()
    y = f + sigma_sq**0.5 * np.random.standard_normal(f.shape)
    F_post = GPR(k_true, alpha=sigma_sq).fit(X[i_tr], y[i_tr])
    _, sd = F_post.predict(X, return_std=True)
    V = sd**2  # True posterior variance.
    # Compute bound with C_hat, true sigma0_sq and true sigma_sq.
    gpb = GPBound(C=C_hat * np.eye(1), sigma0_sq=sigma0_sq_hat, sigma_sq=sigma_sq)
    gpb.fit(X_tr)
    V_hat = gpb.var(X)
    return {
        "V": V,
        "V_hat": V_hat,
        "kernel": kernel,
        "ls": ls,
        "C": C,
        "C_hat": C_hat,
        "sigma0_sq": sigma0_sq,
        "sigma0_sq_hat": sigma0_sq_hat,
    }


def run_directionality(
    N_tr,
    N_te,
    D,
    C,
    C_hat=None,
    sigma0_sq=1.0,
    sigma_sq=1e-8,
    sd_tr=None,
    range_te=(0, np.pi),
    dims_te=None,
):
    sd_tr = 1 / D**0.5 if sd_tr is None else sd_tr
    C_hat = C if C_hat is None else C_hat
    X_tr = sd_tr * np.random.standard_normal((N_tr, D))
    data = [X_tr]
    for i in dims_te:
        X_te_i = np.zeros((N_te, D))
        X_te_i[:, i] = np.linspace(range_te[0], range_te[1], N_te)
        data.append(X_te_i)
    X = np.row_stack(data)
    i_tr = range(N_tr)
    ids = np.repeat(range(1 + len(dims_te)), np.repeat([N_tr, N_te], [1, len(dims_te)]))

    ls = (sigma0_sq / C) ** 0.5
    kernel_true = RBF(ls, "fixed")
    f = GPR(kernel_true).sample_y(X, random_state=None).ravel()
    y = f + sigma_sq**0.5 * np.random.standard_normal(f.shape)

    sigma_sq_fhat = 1e-6
    f_hat = GPR(RBF(ls, "fixed"), alpha=sigma_sq_fhat).fit(X[i_tr], y[i_tr])
    y_hat = f_hat.predict(X)

    gpb = GPBound(C=C_hat * np.eye(D), sigma0_sq=sigma0_sq, sigma_sq=sigma_sq)
    gpb.fit(X[i_tr], y[i_tr])
    V_hat = gpb.var(X)
    B_hat = gpb.bias(X, y_hat, version="exact")

    F_post = GPR(kernel_true, alpha=sigma_sq).fit(X[i_tr], y[i_tr])
    f_post_mean, sd = F_post.predict(X, return_std=True)
    B = y_hat - f_post_mean
    V = sd**2
    return {
        "d": (X**2).sum(axis=1) ** 0.5,
        "V": V,
        "V_hat": V_hat,
        "B": B,
        "B_hat": B_hat,
        "id": ids,
    }


def plot_example_2d(file, width, height):
    n_tr, n_te = 20, 30
    X, y, yhat, i_tr, i_te1, i_te2 = run_example_2d(n_tr=n_tr, n_te=n_te)
    yg = y[~i_tr].reshape(n_te, n_te)
    x1g = X[~i_tr, 0].reshape(n_te, n_te)
    x2g = X[~i_tr, 1].reshape(n_te, n_te)
    x_te1 = X[i_te1][-1]
    x_te2 = X[i_te2][-1]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(width, height))
    ax.plot_surface(x1g, x2g, yg)
    ax.view_init(elev=10, azim=70)
    ax.scatter(X[i_tr, 0], X[i_tr, 1], zs=y.min(), c="black", s=0.5)
    ax.scatter(x_te1[0], x_te1[1], zs=y.min(), c="blue", marker="*")
    ax.scatter(x_te2[0], x_te2[1], zs=y.min(), c="orange", marker="*")
    ax.set(
        xticklabels=[],
        yticklabels=[],
        zticklabels=[],
        xticks=[],
        yticks=[],
        zticks=[],
        xlabel="$x_1$",
        ylabel="$x_2$",
        zlabel="$y$",
    )
    ax.set_xlabel("$x_1$", labelpad=-10)
    ax.set_ylabel("$x_2$", labelpad=-10)
    ax.set_zlabel("$y$", labelpad=-10)
    ax.set_box_aspect([2, 1, 1.5])
    plt.tight_layout()
    if file:
        plt.savefig(file, dpi=500)
    else:
        plt.show()


def plot_example_2d_2(file, width, height):
    n_tr, n_te = 20, 30
    X, y, yhat, i_tr, i_te1, i_te2 = run_example_2d(n_tr=n_tr, n_te=n_te)
    error_te1 = (y[i_te1] - yhat[i_te1]) ** 2
    error_te2 = (y[i_te2] - yhat[i_te2]) ** 2
    x_te1 = X[i_te1][-1]
    x_te2 = X[i_te2][-1]
    fig, axs = plt.subplots(2, 1, figsize=(width, height))
    ax = axs[0]
    ax.scatter(X[i_tr, 0], X[i_tr, 1], c="black", label="Train data", s=0.5)
    ax.scatter(
        x_te1[0],
        x_te1[1],
        c="blue",
        label="Test in $x_1$",
        marker="*",
    )
    ax.scatter(
        x_te2[0],
        x_te2[1],
        c="orange",
        label="Test in $x_2$",
        marker="*",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(fontsize="8")

    ax = axs[1]
    ax.plot(error_te1, label="$x_1$ direction", c="blue")
    ax.scatter(len(error_te1) - 1, error_te1[-1], c="blue", marker="*")
    ax.plot(error_te2, label="$x_2$ direction", c="orange")
    ax.scatter(len(error_te2) - 1, error_te2[-1], c="orange", marker="*")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Distance")
    ax.set_ylabel("Error")
    ax.legend(fontsize="8")
    plt.tight_layout()
    if file:
        plt.savefig(file, dpi=500)
    else:
        plt.show()


def run_example_2d(n_tr=20, n_te=30):
    np.random.seed(2023)
    X_tr = np.random.normal(0, 0.2, (n_tr, 2))
    x1 = np.linspace(-3, 3, n_te)
    x2 = np.linspace(-3, 3, n_te)
    x1g, x2g = np.meshgrid(x1, x2)
    X_te = np.hstack([x1g.reshape(-1, 1), x2g.reshape(-1, 1)])
    X = np.vstack([X_tr, X_te])
    y = GPR(RBF((0.3, 10))).sample_y(X, random_state=1).ravel()

    i_tr = np.repeat([True, False], [n_tr, len(X) - n_tr])
    i_te1 = X[:, 1] == x2[n_te // 2]
    i_te2 = X[:, 0] == x1[n_te // 2]
    fhat = RandomForestRegressor().fit(X[i_tr], y[i_tr])
    yhat = fhat.predict(X)
    return X, y, yhat, i_tr, i_te1, i_te2


def plot_toy(file, width, height):
    x0 = 0
    x_tr, y_tr, x_te, f = make_toy(N_tr=50, x0=0, seed=2023)
    estimator = SVR(C=20, epsilon=0.01).fit(x_tr, y_tr)
    f_hat = estimator.predict
    xmin = np.min([np.min(x_tr), np.min(x_te)])
    xmax = np.max([np.max(x_tr), np.max(x_te)])
    x_grid = np.linspace(xmin, xmax, 200)[:, None]

    _, ax = plt.subplots(figsize=(width, height))
    ax.plot(x_grid, f(x_grid), c="black", label="$f(x)$")
    ax.plot(x_grid, f_hat(x_grid), c="blue", ls="dashed", label="$\hat f(x)$")
    ax.scatter(x_tr, y_tr, c="grey", s=10, label="tr")
    ax.scatter(x_te, f(x_te), c="red", label="te", marker="*", s=100)
    ax.legend()
    plt.box(None)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if file:
        plt.savefig(file, dpi=500)
    else:
        plt.show()


def make_toy(N_tr=200, x0=0, seed=2023):
    np.random.seed(seed)

    def f(x, x0=x0):
        return np.where(
            x < x0,
            0.5 * np.sin(0.5 * x),
            0.5 * np.sin(4 * x),
        ).ravel()

    x_tr = np.linspace(-2, 2, N_tr)[:, None]
    x_te = np.array([-3, 3])[:, None]
    y_tr = f(x_tr) + 0.05 * np.random.standard_normal(x_tr.shape[0])

    return x_tr, y_tr, x_te, f


if __name__ == "__main__":
    from experiments.utils import load_config

    config = load_config(Path(__file__).parent.parent / "config.json")
    DIR_FIGURES = config["dir_figures"]

    plot_kernels(Path(DIR_FIGURES) / "kernels.pdf", height=2, width=4)
    plot_kernel_variances_1d(
        Path(DIR_FIGURES) / "kernels_variance.pdf", height=2, width=4
    )
    plot_directionality(Path(DIR_FIGURES) / "directionality.pdf", height=2, width=4)
    plot_example_2d(Path(DIR_FIGURES) / "example_2d.pdf", height=3, width=3)
    plot_example_2d_2(Path(DIR_FIGURES) / "example_2d_2.pdf", height=4, width=2)
    plot_toy(Path(DIR_FIGURES) / "toy1d.pdf", height=5, width=10)

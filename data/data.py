from pathlib import Path
from typing import Literal
from urllib.request import urlretrieve
import openml
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.datasets import fetch_california_housing
import requests
from zipfile import ZipFile
import numpy as np
from scipy.sparse import csr_matrix
import ase.io


def get_dataset(dataset: str, DIR_DATA, return_names=False, scale_=True):
    match dataset:
        case "autompg":
            return get_autompg(DIR_DATA, return_names=return_names, scale_=scale_)
        case "winequality":
            return get_winequality(DIR_DATA, return_names=return_names, scale_=scale_)
        case "airquality":
            return get_airquality(DIR_DATA, return_names=return_names, scale_=scale_)
        case "airfoil":
            return get_airfoil(DIR_DATA, return_names=return_names, scale_=scale_)
        case "yearpredictionmsd":
            return get_yearpredictionmsd(
                DIR_DATA, return_names=return_names, scale_=scale_
            )
        case "qm9":
            return get_qm9(DIR_DATA, return_names=return_names, scale_=scale_)
        case "oe62":
            return get_oe62(DIR_DATA, return_names=return_names, scale_=scale_)
        case "aa":
            return get_aa(DIR_DATA, return_names=return_names, scale_=scale_)
        case "abalone":
            return get_abalone(DIR_DATA, return_names=return_names, scale_=scale_)
        case "california":
            return get_california(DIR_DATA, return_names=return_names, scale_=scale_)
        case "concrete":
            return get_concrete(DIR_DATA, return_names=return_names, scale_=scale_)
        case "cpu_small":
            return get_cpu_small(DIR_DATA, return_names=return_names, scale_=scale_)
        case "superconductor":
            return get_superconductor(
                DIR_DATA, return_names=return_names, scale_=scale_
            )
        case "winequality-red":
            return get_winequality(
                DIR_DATA, return_names=return_names, scale_=scale_, color="red"
            )
        case "winequality-white":
            return get_winequality(
                DIR_DATA, return_names=return_names, scale_=scale_, color="white"
            )


def get_openml_dataset(id, cache_dir):
    if cache_dir is not None:
        openml.config.cache_directory = cache_dir
    return openml.datasets.get_dataset(id)


def get_autompg(cache_dir=None, scale_=True, return_names=False):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    id = 196
    data = get_openml_dataset(id, cache_dir)
    X, y, _, attribute_names = data.get_data(target=data.default_target_attribute)
    not_na = ~X.isna().any(axis=1)
    X = X[not_na]
    y = y[not_na]
    X = pd.get_dummies(X, columns=["origin"])
    attribute_names = X.columns.to_list()
    X = X.apply(pd.to_numeric).values
    y = y.values
    if scale_:
        X = scale(X)
        y = scale(y)
    if return_names:
        return X, y, attribute_names
    else:
        return X, y


def get_winequality(cache_dir=None, scale_=True, return_names=False, color="both"):
    url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    id = 287
    data = get_openml_dataset(id, cache_dir)
    X, y, _, attribute_names = data.get_data(target=data.default_target_attribute)
    n_red = 1599
    if color == "red":
        i = range(n_red)
    elif color == "white":
        i = range(n_red, X.shape[0])
    else:
        i = range(X.shape[0])
        X["color"] = pd.Categorical(
            np.repeat(["red", "white"], [n_red, len(X) - n_red])
        )
        X = pd.get_dummies(X, columns=["color"])
        attribute_names = X.columns.to_list()
    X = X.values[i]
    y = y.values[i]
    if scale_:
        X = scale(X)
        y = scale(y)
    if return_names:
        return X, y, attribute_names
    else:
        return X, y


def get_airquality(cache_dir: Path = None, scale_=True, return_names=False):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
    filename = "AirQualityUCI.zip"
    if not (cache_dir / filename).exists():
        r = requests.get(url)
        if r.status_code == 200:
            with open(cache_dir / filename, "wb") as f:
                f.write(r.content)
    with ZipFile(cache_dir / filename, "r") as zip:
        with zip.open("AirQualityUCI.csv") as f:
            data = pd.read_csv(f, sep=";", decimal=",")
    # Cleaning copied from https://bitbucket.org/edahelsinki/regressionfunction/src/master/python_notebooks/regressionfunction_notebook.ipynb
    data = data.replace(-200, np.nan)
    # impute cases where only 1 hour of data is missing by the mean of its successor and predessor
    for j in range(data.shape[1]):
        for i in range(1, data.shape[0]):
            if (
                (pd.isna(data.iloc[i, j]))
                and not pd.isna(data.iloc[i - 1, j])
                and not pd.isna(data.iloc[i + 1, j])
            ):
                data.iloc[i, j] = (data.iloc[i - 1, j] + data.iloc[i + 1, j]) / 2
    data = data.drop(columns=["NMHC(GT)"])  # Mostly NA.
    data = data.dropna(axis=1, how="all").dropna(axis=0)
    covariates = [
        "PT08.S1(CO)",
        "C6H6(GT)",
        "PT08.S2(NMHC)",
        "NOx(GT)",
        "PT08.S3(NOx)",
        "NO2(GT)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        "T",
        "RH",
        "AH",
    ]
    target = "CO(GT)"
    X = data.loc[:, covariates]
    y = data.loc[:, target]
    X = X.values
    y = y.values
    if scale_:
        X = scale(X)
        y = scale(y)
    if return_names:
        return X, y, covariates
    else:
        return X, y


def get_airfoil(cache_dir=None, scale_=True, return_names=False):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
    id = 43919
    data = get_openml_dataset(id, cache_dir)
    X, y, _, attribute_names = data.get_data(target=data.default_target_attribute)
    X = X.values
    y = y.values
    if scale_:
        X = scale(X)
        y = scale(y)
    if return_names:
        return X, y, attribute_names
    else:
        return X, y


def get_yearpredictionmsd(cache_dir=None, scale_=True, return_names=False, N=10000):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"
    id = 44027  # Processed version.
    data = get_openml_dataset(id, cache_dir)
    X, y, _, attribute_names = data.get_data(target=data.default_target_attribute)
    X, y = X[:N], y[:N]
    X = X.values
    y = y.values
    if scale_:
        X = scale(X)
        y = scale(y)
    if return_names:
        attribute_names = [f"avg{i:02d}" for i in range(1, 13)] + [
            f"cov{i:02d}" for i in range(1, 79)
        ]
        return X, y, attribute_names
    else:
        return X, y


def get_qm9(
    cache_dir: Path = None, scale_=True, return_names=False, n_components=30, N=10000
):
    """Read a subset of the QM9 dataset
    Subset: N heaviest molecules and n_components PCA components of MBTR descriptors."""
    path_data = cache_dir / "QM9-processed.npz"
    if not (path_data).exists():
        mbtr, homo, molecules = get_molecular_raw(cache_dir, "QM9")
        mbtr_pca = PCA(n_components, whiten=True).fit_transform(scale(mbtr))
        mol_weights = np.array([mol.get_masses().sum() for mol in molecules])
        i_weights = np.argsort(mol_weights)
        i_subset = i_weights[-N:]
        np.savez(path_data, X=mbtr_pca[i_subset], y=homo[i_subset])
    with np.load(path_data) as file:
        X, y = file["X"], file["y"]

    if scale_:
        X = scale(X)
        y = scale(y)
    if return_names:
        attribute_names = [f"PC{i}" for i in range(1, X.shape[1] + 1)]
        return X, y, attribute_names
    else:
        return X, y


def get_oe62(
    cache_dir: Path = None, scale_=True, return_names=False, n_components=30, N=10000
):
    """Read a subset of the OE62 dataset
    Subset: N heaviest molecules and n_components PCA components of MBTR descriptors."""
    path_data = cache_dir / "OE62-processed.npz"
    if not (path_data).exists():
        mbtr, homo, molecules = get_molecular_raw(cache_dir, "OE62")
        mbtr_pca = PCA(n_components, whiten=True).fit_transform(scale(mbtr))
        mol_weights = np.array([mol.get_masses().sum() for mol in molecules])
        i_weights = np.argsort(mol_weights)
        i_subset = i_weights[-N:]
        np.savez(path_data, X=mbtr_pca[i_subset], y=homo[i_subset])
    with np.load(path_data) as file:
        X, y = file["X"], file["y"]

    if scale_:
        X = scale(X)
        y = scale(y)
    if return_names:
        attribute_names = [f"PC{i}" for i in range(1, X.shape[1] + 1)]
        return X, y, attribute_names
    else:
        return X, y


def get_aa(
    cache_dir: Path = None, scale_=True, return_names=False, n_components=30, N=10000
):
    """Read a subset of the AA dataset
    Subset: N heaviest molecules and n_components PCA components of MBTR descriptors."""
    path_data = cache_dir / "AA-processed.npz"
    if not (path_data).exists():
        mbtr, homo, molecules = get_molecular_raw(cache_dir, "AA")
        mbtr_pca = PCA(n_components, whiten=True).fit_transform(scale(mbtr))
        mol_weights = np.array([mol.get_masses().sum() for mol in molecules])
        i_weights = np.argsort(mol_weights)
        i_subset = i_weights[-N:]
        np.savez(path_data, X=mbtr_pca[i_subset], y=homo[i_subset])
    with np.load(path_data) as file:
        X, y = file["X"], file["y"]

    if scale_:
        X = scale(X)
        y = scale(y)
    if return_names:
        attribute_names = [f"PC{i}" for i in range(1, X.shape[1] + 1)]
        return X, y, attribute_names
    else:
        return X, y


def get_molecular_raw(
    cache_dir: Path = None, dataset: Literal["AA", "OE62", "QM9"] = "QM9"
):
    if dataset == "OE62":
        url_xyz = "https://zenodo.org/record/4035923/files/data.xyz"
        url_mbtr = "https://zenodo.org/record/4035923/files/mbtr_0.02.npz"
        url_homo = "https://zenodo.org/record/4035923/files/HOMO.txt"
    elif dataset == "QM9":
        url_xyz = "https://zenodo.org/record/4035918/files/data.xyz"
        url_mbtr = "https://zenodo.org/record/4035918/files/mbtr_0.1.npz"
        url_homo = "https://zenodo.org/record/4035918/files/HOMO.txt"
    elif dataset == "AA":
        url_xyz = "https://zenodo.org/record/5872941/files/data.xyz"
        url_mbtr = "https://zenodo.org/record/3967308/files/mbtr_k2.npz"
        url_homo = "https://zenodo.org/record/5872941/files/HOMO.txt"
    else:
        raise Exception(f"Unknown dataset '{dataset}'")
    path_homo = cache_dir / f"{dataset}-HOMO.txt"
    if not path_homo.exists():
        print(f"Retrieving {url_homo} to {path_homo}.")
        urlretrieve(url_homo, path_homo)

    path_mbtr = cache_dir / f"{dataset}-MBTR.npz"
    if not path_mbtr.exists():
        print(f"Retrieving {url_mbtr} to {path_mbtr}.")
        urlretrieve(url_mbtr, path_mbtr)

    path_xyz = cache_dir / f"{dataset}.xyz"
    if not path_xyz.exists():
        print(f"Retrieving {url_xyz} to {path_xyz}.")
        urlretrieve(url_xyz, path_xyz)

    homo = np.loadtxt(path_homo)
    with np.load(path_mbtr) as loader:
        mbtr = csr_matrix(
            (loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"]
        ).toarray()
    with open(path_xyz, "r") as xyz_file:
        molecules = [x for x in ase.io.iread(xyz_file)]

    return mbtr, homo, molecules


def get_abalone(cache_dir=None, scale_=True, return_names=False):
    id = 183
    data = get_openml_dataset(id, cache_dir)
    X, y, _, attribute_names = data.get_data(target=data.default_target_attribute)
    X = pd.get_dummies(X, columns=["Sex"])
    attribute_names = X.columns.tolist()
    X = X.apply(pd.to_numeric).values
    y = pd.to_numeric(y).to_numpy()
    if scale_:
        X = scale(X)
        y = scale(y)
    if return_names:
        return X, y, attribute_names
    else:
        return X, y


def get_california(cache_dir=None, scale_=True, return_names=False):
    data = fetch_california_housing(data_home=cache_dir)
    X, y, feature_names = [data[k] for k in ["data", "target", "feature_names"]]
    if scale_:
        X = scale(X)
        y = scale(y)
    if return_names:
        return X, y, feature_names
    else:
        return X, y


def get_concrete(cache_dir=None, scale_=True, return_names=False):
    id = 4353
    data = get_openml_dataset(id, cache_dir)
    X, _, _, attribute_names = data.get_data()
    y = X.pop("Concrete compressive strength(MPa. megapascals)")
    attribute_names = X.columns.to_list()
    X = X.apply(pd.to_numeric).values
    y = y.values
    if scale_:
        X = scale(X)
        y = scale(y)
    if return_names:
        return X, y, attribute_names
    else:
        return X, y


def get_cpu_small(cache_dir=None, scale_=True, return_names=False):
    id = 562
    data = get_openml_dataset(id, cache_dir)
    X, y, _, attribute_names = data.get_data(target=data.default_target_attribute)
    X = X.apply(pd.to_numeric).values
    y = y.values
    if scale_:
        X = scale(X)
        y = scale(y)
    if return_names:
        return X, y, attribute_names
    else:
        return X, y


def get_superconductor(cache_dir=None, scale_=True, return_names=False):
    id = 43174
    data = get_openml_dataset(id, cache_dir)
    X, y, _, attribute_names = data.get_data(target=data.default_target_attribute)
    X = X.apply(pd.to_numeric).values
    y = y.values
    if scale_:
        X = scale(X)
        y = scale(y)
    if return_names:
        return X, y, attribute_names
    else:
        return X, y


if __name__ == "__main__":
    DIR_DATA = Path(__file__).parent

    get_autompg(DIR_DATA)
    get_winequality(DIR_DATA)
    get_airquality(DIR_DATA)
    get_airfoil(DIR_DATA)
    get_yearpredictionmsd(DIR_DATA)
    get_qm9(DIR_DATA)
    get_oe62(DIR_DATA)
    get_aa(DIR_DATA)
    get_abalone(DIR_DATA)
    get_california(DIR_DATA)
    get_concrete(DIR_DATA)
    get_cpu_small(DIR_DATA)
    get_superconductor(DIR_DATA)

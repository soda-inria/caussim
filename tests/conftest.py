"""Pytest configuration module"""
import pandas as pd
import numpy as np
import pytest
from caussim.data.causal_df import CausalDf

from caussim.data.loading import load_twins

N = 10
dim = 2


@pytest.fixture(scope="session", name="dummy_factuals")
def make_dummy_factuals(N=10, dim=2):
    X = np.random.rand(N, dim)
    A = np.random.randint(0, dim, N)
    Y = np.random.rand(N)
    return A, X, Y


@pytest.fixture(scope="session", name="dummy_causal_dataset")
def make_dummy_causal_dataset(N=10, dim=2):
    X = pd.DataFrame(np.random.rand(N, dim))
    X.columns = [f"x_{i}" for i in range(len(X.columns))]
    a = np.random.randint(0, dim, N)
    mu_1 = np.random.rand(N)
    mu_0 = np.random.rand(N)
    e = np.random.rand(N)
    df = pd.DataFrame({"a": a, "mu_1": mu_1, "mu_0": mu_0, "e": e})
    df["y_1"] = mu_1
    df["y_0"] = mu_0
    df["y"] = df["y_0"] * (1 - df["a"]) + df["a"] * df["y_1"]
    return CausalDf(
        df=pd.concat([df, X], axis=1),
        dataset_name="dummy_df",
        overlap_parameter=1,
        random_state=0,
        dgp=1
    )



@pytest.fixture(scope="session", name="dummy_causal_dataset_100")
def make_dummy_causal_dataset_100(N=100, dim=2):
    X = pd.DataFrame(np.random.rand(N, dim))
    X.columns = [f"x_{i}" for i in range(len(X.columns))]
    a = np.random.randint(0, dim, N)
    mu_1 = np.random.rand(N)
    mu_0 = np.random.rand(N)
    e = np.random.rand(N)
    df = pd.DataFrame({"a": a, "mu_1": mu_1, "mu_0": mu_0, "e": e})
    df["y_1"] = mu_1
    df["y_0"] = mu_0
    df["y"] = df["y_0"] * (1 - df["a"]) + df["a"] * df["y_1"]
    return CausalDf(pd.concat([df, X], axis=1))


@pytest.fixture(scope="session")
def twins_dataset():
    twins_df = load_twins()
    return twins_df

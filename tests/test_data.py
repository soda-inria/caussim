import numpy as np

import pytest
from caussim.config import CAUSAL_DF_COLUMNS
from caussim.data.causal_df import CausalDf

from caussim.data.loading import (
    load_acic_2018_from_ufid,
    load_twins,
    twins_propensity_model,
)
from caussim.data.loading import load_acic_2016
from caussim.pdistances.mmd import total_variation_distance


# some fixtures
@pytest.fixture(scope="module")
def acic_2016_dataset():
    acic_2016_df = load_acic_2016(1, 1)
    return acic_2016_df

@pytest.fixture(scope="module")
def acic_2018_dataset():
    acic_2018_df = load_acic_2018_from_ufid(ufid="a957b431a74a43a0bb7cc52e1c84c8ad")
    return acic_2018_df
# Some utils
def _check_causal_df(causal_df: CausalDf):
    """Common tests for causal dataframes loading
    Args:
        causal_df (CausalDf):
    """
    causal_df_pandas = causal_df.df
    assert set(CAUSAL_DF_COLUMNS).intersection(causal_df_pandas.columns) == set(
        CAUSAL_DF_COLUMNS
    )
    assert np.array_equal(
        causal_df_pandas["y"],
        causal_df_pandas["a"] * causal_df_pandas["y_1"]
        + (1 - causal_df_pandas["a"]) * causal_df_pandas["y_0"],
    )
    assert (causal_df_pandas["e"] <= 1).all() and (causal_df_pandas["e"] >= 0).all()
    assert causal_df_pandas["a"].isin([0, 1]).all()

# Tests
def test_CausalDf_bootstrap(dummy_causal_dataset):
    X = dummy_causal_dataset.get_X()
    bootstrapped_df = dummy_causal_dataset.bootstrap(seed=0)
    bootstrapped_X = bootstrapped_df.get_X()
    assert not np.array_equal(X, bootstrapped_X)
    inclusions =  [row in X for row in bootstrapped_X]
    assert np.sum(inclusions) == len(bootstrapped_X)    


# test simulated data loading
def test_load_twins(
    twins_dataset,
):
    _check_causal_df(twins_dataset)


def test_load_twins_random():
    twins_default = load_twins()
    twins_r_0 = load_twins(random_state=0)

    assert np.array_equal(
        twins_default.df.drop(CAUSAL_DF_COLUMNS, axis=1),
        twins_r_0.df.drop(CAUSAL_DF_COLUMNS, axis=1),
    )
    assert not np.array_equal(
        twins_default.df[CAUSAL_DF_COLUMNS],
        twins_r_0.df[CAUSAL_DF_COLUMNS],
    )

def test_twins_propensity_model(twins_dataset):
    twins_covariates = twins_dataset.df.drop(CAUSAL_DF_COLUMNS, axis=1)
    ps, a = twins_propensity_model(
        twins_covariates=twins_covariates, random_state=0, alpha=1
    )
    assert (ps <= 1).all() and (ps >= 0).all()
    assert set(a) == {0, 1}

# more elabaroted stuf
def test_twins_tv_alpha(twins_dataset):
    twins_covariates = twins_dataset.df.drop(CAUSAL_DF_COLUMNS, axis=1)
    ps, a = twins_propensity_model(
        twins_covariates=twins_covariates, random_state=0, alpha=1
    )
    ps_wo_overlap, a_w_overlap = twins_propensity_model(
        twins_covariates=twins_covariates, random_state=0, alpha=2
    )
    tv = total_variation_distance(ps / a.mean(), (1 - ps) / (1 - a).mean())
    tv_wo_overlap = total_variation_distance(
        ps_wo_overlap / a_w_overlap.mean(),
        (1 - ps_wo_overlap) / (1 - a_w_overlap).mean(),
    )
    assert tv_wo_overlap > tv


def test_load_acic_2018(acic_2018_dataset):
    _check_causal_df(acic_2018_dataset)

def test_load_acic_2016(acic_2016_dataset):
    _check_causal_df(acic_2016_dataset)

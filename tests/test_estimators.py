# %%

import pytest
from caussim.estimation.estimators import (
    AteEstimator,
    IdentityTransformer,
    SFTLearner,
    SLearner,
    TLearner,
    get_treatment_and_covariates,
)
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Ridge


def test_identity_transformer(dummy_factuals):
    A, X_cov, Y = dummy_factuals
    transformer = IdentityTransformer()
    X_transform = transformer.fit_transform(X_cov)
    assert np.array_equal(X_transform, X_cov)


def test_metalearner(dummy_factuals):
    a, X_cov, y = dummy_factuals
    X = np.column_stack((a, X_cov))
    final_estimator = Ridge()
    meta_learner = SLearner(final_estimator)
    meta_learner.fit(X, y)
    y_hat_1 = meta_learner.predict(np.column_stack((np.ones_like(a), X_cov)))
    y_hat_0 = meta_learner.predict(np.column_stack((np.zeros_like(a), X_cov)))
    hat_cate = meta_learner.predict_cate(X)
    assert np.array_equal(y_hat_1 - y_hat_0, hat_cate)


def test_slearner(dummy_factuals):
    a, X_cov, y = dummy_factuals
    covariates_dim = X_cov.shape[1]

    regressor = Ridge()
    featurizer = Nystroem(n_components=covariates_dim, random_state=0)

    meta_learner = SLearner(regressor, featurizer)
    y_hat = meta_learner.fit_predict(np.column_stack((a, X_cov)), y)
    # For Nystroem, test that the components of a given class are indeed in the train set
    assert (
        len(
            set(map(tuple, meta_learner.featurizer_.components_)).intersection(
                set(map(tuple, X_cov))
            )
        )
        == covariates_dim
    )


def test_tlearner(dummy_factuals):
    a, X_cov, y = dummy_factuals
    covariates_dim = X_cov.shape[1]

    regressor = Ridge()
    featurizer = Nystroem(n_components=covariates_dim, random_state=0)
    meta_learner = TLearner(regressor, featurizer=featurizer)
    y_hat = meta_learner.fit_predict(np.column_stack((a, X_cov)), y)
    # test that the components of each popualtion are indeed chosen in the corresponding population training set
    assert (
        len(
            set(map(tuple, meta_learner.featurizer_control_.components_)).intersection(
                set(map(tuple, X_cov[a == 0]))
            )
        )
        == covariates_dim
    )
    assert (
        len(
            set(map(tuple, meta_learner.featurizer_treated_.components_)).intersection(
                set(map(tuple, X_cov[a == 1]))
            )
        )
        == covariates_dim
    )


def test_sftlearner(dummy_factuals):
    a, X_cov, Y = dummy_factuals
    covariates_dim = X_cov.shape[1]
    regressor = Ridge()
    featurizer = Nystroem(n_components=covariates_dim, random_state=0)
    meta_learner = SFTLearner(regressor, featurizer=featurizer)
    Y_hat = meta_learner.fit_predict(np.column_stack((a, X_cov)), Y)

    assert (
        len(
            set(map(tuple, meta_learner.featurizer_.components_)).intersection(
                set(map(tuple, X_cov))
            )
        )
        == covariates_dim
    )


def test_get_treatment_and_covariates(dummy_factuals):
    a, X_cov, _ = dummy_factuals
    XX_bad = np.column_stack((X_cov, a))
    expected_warn_msg = "First column of covariates should contains the treatment indicator as binary values."
    with pytest.raises(ValueError, match=expected_warn_msg):
        get_treatment_and_covariates(XX_bad)
    XX_good = np.column_stack((a, X_cov))
    a_retrieved, X_cov_retrieved = get_treatment_and_covariates(XX_good)
    assert np.array_equal(a_retrieved, a)
    assert np.array_equal(X_cov, X_cov_retrieved)


def test_AteEstimator(dummy_causal_dataset_100):
    y = dummy_causal_dataset_100.get_y()
    aX = dummy_causal_dataset_100.get_aX()
    model_rs = 0
    
    outcome_model = Ridge()
    propensity_model = LogisticRegression()
    ate_estimator = AteEstimator(
        outcome_model=outcome_model,
        propensity_model=propensity_model,
        random_state_cv=model_rs
    )
    predictions, estimates, metrics = ate_estimator.fit_predict_estimate(aX, y)
    
    assert np.array_equal(predictions["y"], y)
"""
Utils for double robust ATE estimations : 

## Two meta-learners are available for response surface fit/predict strategies (no effect on IPW estimation)

- slearner: learn one model on concatenated covariates XX=[X, A]  
- tleaner: learn separate model for both treatment population

## Several ATE estimators are available :

- "DM" : Simple difference in means estimator (does not take into account confoundedness)

- "AIPW": Augmented Inverse Propensity Weighting estimator of the ATE, with cross-validation and binary treatment and outcome. The most basic Doubly Robust Estimator

- "IPW": Inverse Propensity Weighting, with or without cross-validation

- "REG": G-computation, See :cite:`robins_new_1986`

"""
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble._stacking import _BaseStacking
from sklearn.kernel_approximation import Nystroem

from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    RandomizedSearchCV,
    cross_validate,
)

from sklearn.base import BaseEstimator, TransformerMixin
import logging

from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import column_or_1d

from caussim.estimation.scores import print_metrics_binary, print_metrics_regression

# ### MetaLearners ### #
SLEARNER_LABEL = "SLearner"
TLEARNER_LABEL = "TLearner"
SFTLEARNER_LABEL = "SftLearner"
RLEARNER_LABEL = "RLearner"
AVAILABLE_LEARNERS = [SLEARNER_LABEL, TLEARNER_LABEL, SFTLEARNER_LABEL, RLEARNER_LABEL]

# TODO: implement trimming (discard extreme examples)
# TODO: use cross_validate_cv from sklearn instead of own estimation of cv

# ### ATE estimator in the DR fashion ### #
class AteEstimator(object):
    def __init__(
        self,
        outcome_model: BaseEstimator = None,
        propensity_model: BaseEstimator = None,
        tau: str = "AIPW",
        n_splits=5,
        random_state_cv=42,
        clip=1e-4,
        meta_learner: str = SLEARNER_LABEL,
    ) -> None:
        """Initialisation

        Args:
            outcome_model (BaseEstimator, optional): [description]. Defaults to None.
            propensity_model (BaseEstimator, optional): [description]. Defaults to None.
            tau (str, optional): [description]. Defaults to "AIPW".
            nb_splits (int, optional): [description]. Defaults to 5.
            random_state_cv (int, optional): [description]. Defaults to 42.
            min_propensity ([type], optional): Used to trim the high and low propensity scores that violate the overlap assumption. Defaults to 1e-4.
        """
        super().__init__()
        self.tau = tau
        if self.tau in ["DM", "IPW"]:
            outcome_model = None
            logging.info("TAU set to IPW, forcing outcome_model to be None")
        if self.tau in ["DM", "REG"]:
            propensity_model = None
            logging.info("TAU set to REG, forcing propensity_model to be None")
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.meta_learner_type = meta_learner
        self.outcome_models = []
        self.propensity_models = []
        self.random_state_cv = random_state_cv
        self.n_splits = n_splits
        self.clip = clip
        self.meta_learner = set_meta_learner(self.meta_learner_type, self.outcome_model)

        # TODO: should have different kfold for propensity and outcome
        if self.n_splits == 1:
            self.kfold = dummy1Fold()
        else:
            if self.outcome_model is not None:
                if self.outcome_model._estimator_type == "classifier":
                    self.kfold = StratifiedKFold(
                        n_splits=self.n_splits,
                        random_state=self.random_state_cv,
                        shuffle=True,
                    )
                else:
                    self.kfold = KFold(
                        n_splits=self.n_splits,
                        random_state=self.random_state_cv,
                        shuffle=True,
                    )
            else:
                self.kfold = StratifiedKFold(
                    n_splits=self.n_splits,
                    random_state=self.random_state_cv,
                    shuffle=True,
                )

        self.predictions = {
            "hat_mu_1": [],
            "hat_mu_0": [],
            "hat_a_proba": [],
            "y": [],
            "a": [],
        }
        self.metrics = {}
        self.in_sample_cate = None
        self.in_sample_ate = None

    def fit(self, X, y):
        a, X_cov = get_treatment_and_covariates(X)

        if self.outcome_model is not None:
            splitter_y = self.kfold.split(X_cov, a)
            for train_index, _ in splitter_y:
                outcome_estimator_fold = clone(self.meta_learner)
                outcome_estimator_fold.fit(X[train_index], y[train_index])
                self.outcome_models.append(outcome_estimator_fold)

        if self.propensity_model is not None:
            self._fit_propensity(X_cov, a)

    def _fit_propensity(self, X, a):
        # TODO: add recalibration here with a train/test split
        splitter_a = self.kfold.split(X, a)
        # a = a.reshape(-1, 1)
        for train_index, _ in splitter_a:
            if self.propensity_model is not None:
                propensity_estimator_fold = clone(self.propensity_model, safe=True)
                propensity_estimator_fold.fit(X[train_index], a[train_index])
                self.propensity_models.append(propensity_estimator_fold)

    # TODO: scoring is extern to a model in scikitlearn api and there is no reason to act differently for causal inference.
    def predict(
        self,
        X,
        y,
        leftout: bool = False,
    ) -> Tuple[Dict, Dict]:
        """Run the predictions for the nuisance models.
        If one or both of the nuisance models are set to none, do nothing but return dummy predictions proba = (-1, -1) and empty metrics.

        Args:
            X ([type]): Covariates
            y ([type]): Binary Outcome
            a ([type]): Binary Intervention
            leftout (binary, optional): If True, average the results over the folds and models. Defaults to False.
        Returns:
            predictions [Dict]: [description]
            metrics [Dict]: [description]
        """
        # Forcing one dimension for y and a
        y = y.ravel()
        a, X_cov = get_treatment_and_covariates(X)
        # a = a.ravel()
        # checking wich nuisance models are used
        n_outcome_models = len(self.outcome_models)
        n_ps_models = len(self.propensity_models)
        n_models = max(n_outcome_models, n_ps_models)

        if not leftout:
            chunks_indices = [
                np.arange(len(X))[test_ix] for _, test_ix in self.kfold.split(X_cov, a)
            ]
            chunks_X_cov = [X_cov[test_ix] for _, test_ix in self.kfold.split(X_cov, a)]
            chunks_y = [
                np.array(y)[test_ix] for _, test_ix in self.kfold.split(X_cov, a)
            ]
            chunks_a = [
                np.array(a)[test_ix] for _, test_ix in self.kfold.split(X_cov, a)
            ]
            # if no folds is provided, this is test data, we average over the cv models
        else:
            chunks_indices = [np.arange(len(X)) for i in range(n_models)]
            chunks_X_cov = [X_cov for i in range(n_models)]
            chunks_y = [np.array(y) for i in range(n_models)]
            chunks_a = [np.array(a) for i in range(n_models)]
        mu_1 = []
        mu_0 = []
        a_hat_proba = []
        for i in range(n_models):
            # reshaped_a = np.expand_dims(, axis=1)
            if n_outcome_models > 0:
                if self.outcome_model._estimator_type == "classifier":
                    chunk_y_hat_0 = self.outcome_models[i].predict_proba(
                        np.column_stack([np.zeros_like(chunks_a[i]), chunks_X_cov[i]])
                    )[:, 1]
                    chunk_y_hat_1 = self.outcome_models[i].predict_proba(
                        np.column_stack([np.ones_like(chunks_a[i]), chunks_X_cov[i]])
                    )[:, 1]
                else:
                    chunk_y_hat_0 = self.outcome_models[i].predict(
                        np.column_stack([np.zeros_like(chunks_a[i]), chunks_X_cov[i]])
                    )
                    chunk_y_hat_1 = self.outcome_models[i].predict(
                        np.column_stack([np.ones_like(chunks_a[i]), chunks_X_cov[i]])
                    )
                mu_1.append(chunk_y_hat_1)
                mu_0.append(chunk_y_hat_0)
            if n_ps_models > 0:
                a_hat_proba.append(
                    self.propensity_models[i].predict_proba(chunks_X_cov[i])[:, 1]
                )
        # concatenate the results
        if not leftout:
            if n_outcome_models > 0:
                mu_1 = np.concatenate(mu_1, axis=0)
                mu_0 = np.concatenate(mu_0, axis=0)
            else:
                mu_1, mu_0 = (
                    np.repeat([np.nan], len(y), axis=0),
                    np.repeat([np.nan], len(y), axis=0),
                )
            if n_ps_models > 0:
                a_hat_proba = np.concatenate(a_hat_proba, axis=0)
            else:
                a_hat_proba = np.repeat([np.nan], len(a), axis=0)
            y = np.concatenate(chunks_y, axis=0)
            a = np.concatenate(chunks_a, axis=0)
            indices = np.concatenate(chunks_indices, axis=0)
        else:
            if n_outcome_models > 0:
                mu_1 = np.array(mu_1).mean(axis=0)
                mu_0 = np.array(mu_0).mean(axis=0)
            else:
                mu_1, mu_0 = (
                    np.repeat([np.nan], len(y), axis=0),
                    np.repeat([np.nan], len(y), axis=0),
                )
            if n_ps_models > 0:
                a_hat_proba = np.array(a_hat_proba).mean(axis=0)
            else:
                a_hat_proba = np.repeat([np.nan], len(a), axis=0)
            y = chunks_y[0]
            a = chunks_a[0]
            indices = chunks_indices[0]

        predictions = (
            pd.DataFrame(
                {
                    "idx": indices,
                    "hat_mu_1": mu_1,
                    "hat_mu_0": mu_0,
                    "hat_a_proba": a_hat_proba,
                    "y": y,
                    "a": a,
                }
            )
            .sort_values("idx")
            .reset_index(drop=True)
        )

        # metrics
        metrics = {}
        if self.outcome_model is not None:
            y_hat = a * mu_1 + (1 - a) * mu_0 * (1 - a)
            if self.outcome_model._estimator_type == "classifier":
                outcome_metrics = print_metrics_binary(y, y_hat, verbose=0)
            else:
                outcome_metrics = print_metrics_regression(y, y_hat, verbose=0)
            for metric_name, metric in outcome_metrics.items():
                metrics[f"outcome_{metric_name}"] = metric
        if self.propensity_model is not None:
            if np.all(a == 0) or np.all(a == 1):
                Warning(
                    "All propensity scores are 0 or 1, cannot compute scores for propensity models, returning dummy scores"
                )
                a = np.random.randint(0, 2, size=len(a))
            propensity_metrics = print_metrics_binary(a, a_hat_proba, verbose=0)
            for metric_name, metric in propensity_metrics.items():
                metrics[f"propensity_{metric_name}"] = metric

        # update if we are predicting the insample data
        if not leftout:
            self.predictions = predictions.copy()
            self.metrics = metrics.copy()
        return predictions, metrics

    def estimate(self, predictions, clip=None) -> float:
        if clip is None:
            clip = self.clip
        y = predictions["y"]
        mu_1 = predictions["hat_mu_1"]
        mu_0 = predictions["hat_mu_0"]
        a = predictions["a"]
        # clipping at inference time
        a_hat_proba = np.clip(predictions["hat_a_proba"], clip, 1 - clip)
        if self.tau == "AIPW":
            estimate = tau_aipw(
                y=y,
                mu_1=mu_1,
                mu_0=mu_0,
                a=a,
                a_hat_proba=a_hat_proba,
            )
        elif self.tau == "DM":
            estimate = tau_diff_means(y, a)
        elif self.tau == "IPW":
            estimate = tau_ipw(y=y, a=a, a_hat_proba=a_hat_proba)
        elif self.tau == "REG":
            estimate = tau_reg(mu_1=mu_1, mu_0=mu_0)
        else:
            raise ValueError(f"{self.tau} is not a valid ATE estimator")
        return estimate

    def predict_estimate(self, X, y, leftout=False):
        predictions, metrics = self.predict(X, y, leftout=leftout)
        estimate = self.estimate(predictions)
        return predictions, estimate, metrics

    def fit_predict(self, X, y):
        self.fit(X, y)
        predictions, _, metrics = self.predict_estimate(X, y)
        return predictions, metrics

    def fit_predict_estimate(self, X, y) -> Tuple[Dict, Dict, Dict]:
        """
        - Fit the nuisance models
        - Predict the nuisance parameters
        - Estimate the causal target (ie. estimand)

        Args:
            X ([type]): Covariates
            y ([type]): Target (binary or continuous)
            a ([type]): Intervention (ie. Treatment)

        Returns:
            Tuple[Dict, Dict, Dict]: [predictions, estimate, metrics]
        """
        self.fit(X, y)
        predictions, estimate, metrics = self.predict_estimate(X, y)
        estimate = self.estimate(predictions)
        self.in_sample_cate = estimate["hat_cate"]
        self.in_sample_ate = estimate["hat_ate"]
        return predictions, estimate, metrics


# TODO: make more general by allowing no y estimator if we are not interested in R-risks.
class CateEstimator(object):
    """Estimator class for CATE model. Supports SLearner, SFTLearner, TLearner.

    Args:
        object ([type]): [description]
    """

    def __init__(
        self,
        meta_learner: BaseEstimator,
        a_estimator: BaseEstimator = None,
        y_estimator: BaseEstimator = None,
        a_hyperparameters: Dict = None,
        y_hyperparameters: Dict = None,
        a_scoring: str = "roc_auc",
        y_scoring: str = "r2",
        n_iter: int = 10,
        cv: int = None,
        random_state_hp_search=0,
        n_jobs=-1,
        strict_overlap=1e-10,
    ) -> None:
        self.meta_learner = meta_learner
        self.a_estimator = a_estimator
        self.y_estimator = y_estimator
        if self.y_estimator is not None:
            self.y_hyperparameters = y_hyperparameters
            assert (
                self.y_estimator._estimator_type == "regressor"
            ), "Mean outcome estimator must be a regressor"
        else:
            self.y_hyperparameters = None
            if self.y_hyperparameters is not None:
                logging.warning(
                    "No mean outcome estimator provided, forcing y_hyperparameters to None"
                )
        if self.a_estimator is not None:
            self.a_hyperparameters = a_hyperparameters
            assert (
                self.a_estimator._estimator_type == "classifier"
            ), "Treatment estimator must be a classifier"
        else:
            self.a_hyperparameters = None
            if self.a_hyperparameters is not None:
                logging.warning(
                    "No treatment estimator provided, forcing a_hyperparameters to None"
                )

        if cv is None:
            self.cv = StratifiedKFold(n_splits=5)
        elif cv == 1:
            self.cv = dummy1Fold()
        else:
            self.cv = cv
        self.random_state_hp_search = random_state_hp_search
        self.n_jobs = n_jobs
        self.strict_overlap = strict_overlap
        self.y_scoring = y_scoring
        self.a_scoring = a_scoring
        self.n_iter = n_iter
        assert (
            self.meta_learner.final_estimator._estimator_type == "regressor"
        ), "CATE estimator must be a regressor"

    def fit_nuisances(self, X, y):
        """Learn unknown nuisance ($\\check e$, $\\check m$) necessary for:
        - estimation (R meta-learning)
        - model selection ($\widehat{\mu\mathrm{-risk}}_{IPTW}(\frac{1}{\check e_a}, f)$)

        Args:
            X ([type]): [description]
            Y ([type]): [description]

        Returns:
            [type]: [description]
        """
        if (self.a_estimator is None) or (self.y_estimator is None):
            raise ValueError(
                "No nuisance estimators provided for CATE, cannot estimate nuisances."
            )
        a, X_cov = get_treatment_and_covariates(X)

        # Find appropriate parameters for nuisance models
        self.y_model_rs_ = RandomizedSearchCV(
            estimator=self.y_estimator,
            param_distributions=self.y_hyperparameters,
            scoring=self.y_scoring,
            n_iter=self.n_iter,
            random_state=self.random_state_hp_search,
            cv=None,
        )
        self.a_model_rs_ = RandomizedSearchCV(
            estimator=self.a_estimator,
            param_distributions=self.a_hyperparameters,
            scoring=self.a_scoring,
            n_iter=self.n_iter,
            n_jobs=self.n_jobs,
            random_state=self.random_state_hp_search,
            cv=None,
        )
        self.y_model_rs_results_ = self.y_model_rs_.fit(X_cov, y)
        self.a_model_rs_results_ = self.a_model_rs_.fit(X_cov, a)
        # Refit best model with CV
        splitter_y = self.cv.split(X_cov, a)
        self.y_nuisance_estimators_cv_ = cross_validate(
            clone(self.y_model_rs_results_.best_estimator_),
            X_cov,
            y,
            cv=splitter_y,
            return_estimator=True,
            scoring="neg_mean_squared_error",
        )

        splitter_a = self.cv.split(X_cov, a)
        self.a_nuisance_estimators_cv_ = cross_validate(
            clone(self.a_model_rs_results_.best_estimator_),
            X_cov,
            a,
            cv=splitter_a,
            return_estimator=True,
            scoring="neg_brier_score",
        )
        self.a_nuisance_estimators_ = self.a_nuisance_estimators_cv_["estimator"]
        self.y_nuisance_estimators_ = self.y_nuisance_estimators_cv_["estimator"]
        return self

    def fit(self, X, y):
        # Rq: In case of R-learner, fit should always be done on leftout data (or we should include a nested CV procedure)
        self.meta_learner_ = clone(self.meta_learner)
        self.meta_learner_.fit(X, y)
        return self

    def predict(self, X):
        predictions = {}
        a, X_cov = get_treatment_and_covariates(X)
        if hasattr(self, "a_nuisance_estimators_"):
            hat_e = cross_val_predict_from_fitted(
                estimators=self.a_nuisance_estimators_,
                X=X_cov,
                A=a,
                cv=self.cv,
                method="predict_proba",
            )[:, 1]
            if self.strict_overlap is not None:
                hat_e[hat_e <= 0.5] = hat_e[hat_e <= 0.5] + self.strict_overlap
                hat_e[hat_e > 0.5] = hat_e[hat_e > 0.5] - self.strict_overlap
            predictions["check_e"] = hat_e
        if hasattr(self, "y_nuisance_estimators_"):
            hat_m = cross_val_predict_from_fitted(
                estimators=self.y_nuisance_estimators_,
                X=X_cov,
                A=None,
                method="predict",
                cv=self.cv,
            )
            predictions["check_m"] = hat_m
        if self.meta_learner.__class__.__name__ == RLEARNER_LABEL:
            predictions["hat_tau"] = self.meta_learner_.predict(X)
            predictions["hat_mu_0"] = hat_m - hat_e * predictions["hat_tau"]
            predictions["hat_mu_1"] = predictions["hat_mu_0"] + predictions["hat_tau"]
        else:
            predictions["hat_mu_0"] = self.meta_learner_.predict(
                np.column_stack((np.zeros(X.shape[0]) * 1.0, X_cov))
            )
            predictions["hat_mu_1"] = self.meta_learner_.predict(
                np.column_stack((np.ones(X.shape[0]) * 1.0, X_cov))
            )
            predictions["hat_tau"] = predictions["hat_mu_1"] - predictions["hat_mu_0"]

        return pd.DataFrame(predictions)

    def describe(self):
        model_names = ["Outcome, m", "Treatment, e"]
        model_descriptions = []
        for model_name, model, model_hps in zip(
            model_names,
            [self.y_estimator, self.a_estimator],
            [self.y_hyperparameters, self.a_hyperparameters],
        ):
            model_desc = {"Model": model_name}
            if isinstance(model, _BaseStacking):
                y_stacked_models = []
                for est in model.estimators:
                    if hasattr(est[1], "steps"):
                        y_stacked_models.append(est[1].steps[-1][0])
                    else:
                        y_stacked_models.append(est[0])
                model_desc["Estimator"] = (
                    "StackedRegressor(" + ", ".join(y_stacked_models) + ")"
                )
            else:
                model_desc["Estimator"] = type(model).__name__
            for i, (hp_name, hp_values) in enumerate(model_hps.items()):
                hp_str = (
                    hp_name.replace("_", " ") + ": " + str(hp_values).replace("_", " ")
                )
                if i == 0:
                    model_desc["Hyper-parameters grid"] = hp_str
                else:
                    model_desc = {
                        "Model": "",
                        "Estimator": "",
                        "Hyper-parameters grid": hp_str,
                    }
                model_descriptions.append(model_desc)
        return pd.DataFrame(model_descriptions).set_index(["Model", "Estimator"])


# ### Utils functions ### #
def cross_val_predict_from_fitted(
    estimators, X: np.array, A=None, cv=None, method: str = "predict"
):
    """Compute cross-validation predictions from fitted estimators.

    Args:
        estimators (List): List of fitted estimators.
        X (np.array): Predictors
        splitter ([type], optional): splitter. Defaults to None.
        method (str, optional): estimator method for prediction : ["predict", "predict_proba"]. Defaults to "predict".

    Returns:
        hat_Y (np.array): predictions

    """
    hat_Y = []
    indices = []
    if A is None:
        iterator = dummy1Fold().split(X)
    elif hasattr(cv, "split"):
        iterator = cv.split(X, A)
    for i, (train_ix, test_ix) in enumerate(iterator):
        estimator = estimators[i]
        func = getattr(estimator, method)
        hat_Y.append(func(X[test_ix]))
        indices.append(test_ix)

    if A is None:
        # Average in case of no CV (ie. leftout)
        hat_Y = np.mean(hat_Y, axis=0)
    else:
        indices = np.argsort(np.concatenate(indices, axis=0))
        hat_Y = np.concatenate(hat_Y, axis=0)[indices]
    return hat_Y


### # Utils for meta learners # ###


def get_treatment_and_covariates(X):
    """Split treatment and covariates from full covariate matrix $X=[a, X_cov]$.

    Require that the first column of $X$ is the treatment indicator.

    Args:
        X (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    a = X[:, 0]
    X_cov = X[:, 1:]
    a = column_or_1d(a, warn=True)
    if not np.array_equal(a, a.astype(bool)):
        raise ValueError(
            "First column of covariates should contains the treatment indicator as binary values."
        )
    return a, X_cov


class MetaLearner:
    _required_parameters = ["estimator"]
    """Mixin class for all meta learners."""

    def predict_cate(self, X):
        a, X_cov = get_treatment_and_covariates(X)
        hat_y_1 = self.predict(np.column_stack([np.ones_like(a), X_cov]))
        hat_y_0 = self.predict(np.column_stack([np.zeros_like(a), X_cov]))
        return hat_y_1 - hat_y_0

    def predict(self, X):
        raise NotImplementedError


def set_meta_learner(
    meta_learner_name: str,
    final_estimator: BaseEstimator,
    featurizer: TransformerMixin = None,
):
    if meta_learner_name == TLEARNER_LABEL:
        meta_learner = TLearner(final_estimator, featurizer)
    elif meta_learner_name == SLEARNER_LABEL:
        meta_learner = SLearner(final_estimator, featurizer)
    elif meta_learner_name == SFTLEARNER_LABEL:
        meta_learner = SFTLearner(final_estimator, featurizer)
    elif meta_learner_name == RLEARNER_LABEL:
        meta_learner = make_pipeline(featurizer, final_estimator)
    else:
        raise ValueError(
            "Got {} meta_learner, but supports only following : \n {}".format(
                meta_learner_name, AVAILABLE_LEARNERS
            )
        )
    return meta_learner


class SLearner(MetaLearner, BaseEstimator):
    """Meta-learner with shared featurization and regressors between both treated and control populations.

    Fit/predict a transformation, g of the covariates, then apply a sklearn estimator, f on the transformed covariates augmtented with the treatment variable
    $m(x,a) = f \big([a, g(x)] \big)$
    Args:
        final_estimator (BaseEstimator): _description_
        featurizer (TransformerMixin): _description_
    """

    def __init__(
        self,
        final_estimator: BaseEstimator,
        featurizer: TransformerMixin = None,
    ) -> None:

        self.final_estimator = final_estimator
        if featurizer is None:
            self.featurizer = IdentityTransformer()
        else:
            self.featurizer = featurizer

    def fit(self, X, y):
        a, X_cov = get_treatment_and_covariates(X)
        self.featurizer_ = clone(self.featurizer)
        self.final_estimator_ = clone(self.final_estimator)
        X_transformed = self.featurizer_.fit_transform(X_cov)
        self.final_estimator_.fit(np.column_stack((a, X_transformed)), y)
        return self

    def predict(self, X):
        a, X_cov = get_treatment_and_covariates(X)
        X_transformed = self.featurizer_.transform(X_cov)
        return self.final_estimator_.predict(np.column_stack((a, X_transformed)))

    def predict_proba(self, X):
        a, X_cov = get_treatment_and_covariates(X)
        X_transformed = self.featurizer_.transform(X_cov)
        return self.final_estimator_.predict_proba(np.column_stack((a, X_transformed)))

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class TLearner(MetaLearner, BaseEstimator):
    """Meta-learner with separate featurization and regressors between two populations.

    For each population, fit/predict a transformation, g_a of the covariates, then apply a sklearn estimator, f_a on the transformed covariates:

    $m(x,a) = f_a \big([g_a(x)] \big)$
    Args:
        final_estimator (BaseEstimator): _description_
        featurizer (TransformerMixin, optional): _description_. Defaults to None.
    """

    def __init__(
        self, final_estimator: BaseEstimator, featurizer: TransformerMixin = None
    ) -> None:
        self.final_estimator = final_estimator
        if featurizer is None:
            self.featurizer = IdentityTransformer()
        else:
            self.featurizer = featurizer

    def fit(self, X, y):
        a, X_cov = get_treatment_and_covariates(X)

        self.featurizer_control_ = clone(self.featurizer)
        self.final_estimator_control_ = clone(self.final_estimator)
        self.featurizer_treated_ = clone(self.featurizer)
        self.final_estimator_treated_ = clone(self.final_estimator)

        mask_control = a == 0
        mask_treated = a == 1
        if (mask_control.sum() == 0) or (mask_treated.sum() == 0):
            raise AttributeError(
                "Provided crossfit folds contain training splits that don't contain at least one example of each treatment"
            )
        X_control_transformed = self.featurizer_control_.fit_transform(
            X_cov[mask_control]
        )
        X_treated_transformed = self.featurizer_treated_.fit_transform(
            X_cov[mask_treated]
        )

        self.final_estimator_control_.fit(X_control_transformed, y[mask_control])
        self.final_estimator_treated_.fit(X_treated_transformed, y[mask_treated])

        return self

    def predict(self, X):
        a, X_cov = get_treatment_and_covariates(X)

        mask_control = a == 0
        mask_treated = a == 1
        y = np.empty_like(a) * 0.0

        if sum(mask_control) != 0:
            X_control_transformed = self.featurizer_control_.transform(
                X_cov[mask_control]
            )
            y[mask_control] = self.final_estimator_control_.predict(
                X_control_transformed
            )
        if sum(mask_treated) != 0:
            X_treated_transformed = self.featurizer_treated_.transform(
                X_cov[mask_treated]
            )
            y[mask_treated] = self.final_estimator_treated_.predict(
                X_treated_transformed
            )
        return y

    def predict_proba(self, X):
        a, X_cov = get_treatment_and_covariates(X)

        mask_control = a == 0
        mask_treated = a == 1
        y = np.empty((a.shape[0], 2)) * 0.0

        if sum(mask_control) != 0:
            X_control_transformed = self.featurizer_control_.transform(
                X_cov[mask_control]
            )
            y[mask_control] = self.final_estimator_control_.predict_proba(
                X_control_transformed
            )
        if sum(mask_treated) != 0:
            X_treated_transformed = self.featurizer_treated_.transform(
                X_cov[mask_treated]
            )
            y[mask_treated] = self.final_estimator_treated_.predict_proba(
                X_treated_transformed
            )
        return y

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class SFTLearner(MetaLearner, BaseEstimator):
    """Meta-learner with shared featurization between two populations but separate regressors.
    Fit/predict a shared transformation, g of the covariates, then apply a sklearn estimator, f_a for each population on the transformed covariates:

    $m(x,a) = f_a \big([g(x)] \big)$

    Args:
        final_estimator (BaseEstimator): _description_
        featurizer (TransformerMixin, optional): _description_. Defaults to None.
    """

    def __init__(
        self, final_estimator: BaseEstimator, featurizer: TransformerMixin = None
    ) -> None:
        self.final_estimator = final_estimator
        if featurizer is None:
            self.featurizer = IdentityTransformer()
        else:
            self.featurizer = featurizer

    def fit(self, X, y):
        self.featurizer_ = clone(self.featurizer)
        a, X_cov = get_treatment_and_covariates(X)

        X_transformed = self.featurizer_.fit_transform(X_cov)

        self.final_estimator_control_ = clone(self.final_estimator)
        self.final_estimator_treated_ = clone(self.final_estimator)

        mask_control = a == 0
        mask_treated = a == 1
        if (mask_control.sum() == 0) or (mask_treated.sum() == 0):
            raise AttributeError(
                "Provided crossfit folds contain training splits that don't contain all treatments"
            )

        self.final_estimator_control_.fit(X_transformed[mask_control], y[mask_control])
        self.final_estimator_treated_.fit(X_transformed[mask_treated], y[mask_treated])

        return self

    def predict(self, X):
        a, X_cov = get_treatment_and_covariates(X)
        mask_control = a == 0
        mask_treated = a == 1
        X_transform = self.featurizer_.transform(X_cov)

        y = np.zeros_like(a) * 0.0

        if sum(mask_control) != 0:
            y[mask_control] = self.final_estimator_control_.predict(
                X_transform[mask_control]
            )
        if sum(mask_treated) != 0:
            y[mask_treated] = self.final_estimator_treated_.predict(
                X_transform[mask_treated]
            )
        return y

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class RLearner(MetaLearner, BaseEstimator):
    def __init__(
        self,
        final_estimator: BaseEstimator,
        y_estimator: BaseEstimator,
        a_estimator: BaseEstimator,
        featurizer: TransformerMixin = None,
    ) -> None:
        if featurizer is None:
            self.featurizer = IdentityTransformer()
        else:
            self.featurizer = featurizer
        self.final_estimator = make_pipeline(self.featurizer, final_estimator)
        self.y_estimator = make_pipeline(self.featurizer, y_estimator)
        self.a_estimator = make_pipeline(self.featurizer, a_estimator)

    def fit(self, X, y):
        a, X_cov = get_treatment_and_covariates(X)
        splitter_a = self.cv.split(X_cov, a)
        self.a_estimator_cv_ = cross_validate(
            clone(self.y_estimator),
            X_cov,
            a,
            cv=splitter_a,
            return_estimator=True,
            scoring="neg_brier_score",
        )
        splitter_y = self.cv.split(X_cov, y)
        self.y_estimator_cv_ = cross_validate(
            clone(self.y_estimator),
            X_cov,
            y,
            cv=splitter_y,
            return_estimator=True,
            scoring="neg_brier_score",
        )

        self.hat_e_ = cross_val_predict_from_fitted(
            estimators=self.a_estimator_cv_["estimator"],
            X=X_cov,
            A=a,
            cv=self.cv,
            method="predict_proba",
        )[:, 1]
        self.hat_m_ = cross_val_predict_from_fitted(
            estimators=self.y_estimator_cv_["estimator"],
            X=X,
            A=None,
            cv=self.cv,
            method="predict",
        )
        weights = (a - self.hat_e_) ** 2
        y_tilde = (y - self.hat_m_) / (a - self.hat_e_)
        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(X_cov, y_tilde, regression__sample_weight=weights)

    def predict(self, X):
        _, X_cov = get_treatment_and_covariates(X)
        return self.final_estimator.predict(X_cov)


# TODO: implement X-learner


def get_meta_learner_components(meta_learner: BaseEstimator):
    if hasattr(meta_learner, "featurizer_"):
        components_0 = meta_learner.featurizer_.components_
        components_1 = meta_learner.featurizer_.components_
    elif hasattr(meta_learner, "featurizer_control_"):
        components_0 = meta_learner.featurizer_control_.components_
        components_1 = meta_learner.featurizer_treated_.components_
    return {"components_control": components_0, "components_treated": components_1}


# ### Special Transformers ### #
class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=None):
        if random_state is None:
            self.random_state = 0
        else:
            self.random_state = random_state

    def fit(self, X, y=None):
        self.components_ = None
        return self

    def transform(self, X):
        X = self._validate_data(X, accept_sparse="csr", reset=False)
        return X * 1


class FrozenNystroem(Nystroem, TransformerMixin):
    def __init__(
        self,
        *,
        components=None,
        normalization=None,
        kernel="rbf",
        gamma=None,
        coef0=None,
        degree=None,
        kernel_params=None,
        n_components=100,
        random_state=None,
        n_jobs=None,
    ):
        super().__init__(
            kernel=kernel,
            gamma=gamma,
            coef0=coef0,
            degree=degree,
            kernel_params=kernel_params,
            n_components=n_components,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.components = components
        self.normalization = normalization

    def fit(self, X, y=None):
        if self.components is None:
            super().fit(X, y)
        else:
            self.components_ = self.components
            self.normalization_ = self.normalization
            return self

    def transform(self, X):
        return super().transform(X)


# ### Causal Estimators for ATE ### #
# TODO: add  closed form CI
def tau_diff_means(y, a):
    """Simple difference in means estimator

    Args:
        y ([type]): [description]
        a ([type]): [description]

    Returns:
        [type]: [description]
    """
    y1 = y[a == 1]  # Outcome in treatment grp
    y0 = y[a == 0]  # Outcome in control group
    n1 = a.sum()  # Number of obs in treatment
    n0 = len(a) - n1  # Number of obs in control
    # Difference in means is ATE
    ate = np.mean(y1) - np.mean(y0)
    # 95% Confidence intervals
    se_hat = np.sqrt(np.var(y0) / (n0 - 1) + np.var(y1) / (n1 - 1))
    lower_ci = ate - 1.96 * se_hat
    upper_ci = ate + 1.96 * se_hat
    return {
        "hat_ate": ate,
        "hat_cate": None,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
    }


def tau_ipw(y, a, a_hat_proba):
    cate = a * y / a_hat_proba - (1 - a) * y / (1 - a_hat_proba)
    return {
        "hat_ate": cate.mean(axis=0),
        "hat_cate": None,
        "lower_ci": None,
        "upper_ci": None,
    }


def tau_reg(mu_1, mu_0):
    cate = mu_1 - mu_0
    return {
        "hat_ate": cate.mean(axis=0),
        "hat_cate": cate,
        "lower_ci": None,
        "upper_ci": None,
    }


def aipw_formula(y, mu_1, mu_0, a, a_hat_proba):
    return (
        mu_1
        - mu_0
        + a * (y - mu_1) / a_hat_proba
        - (1 - a) * (y - mu_0) / (1 - a_hat_proba)
    )


def tau_aipw(y, mu_1, mu_0, a, a_hat_proba):
    # NOTE: AIPW does not estimate CATE, HTE should be done with Residual learners (cf. Wager 361)
    cate = aipw_formula(y, mu_1, mu_0, a, a_hat_proba)
    return {
        "hat_ate": cate.mean(axis=0),
        "hat_cate": cate,
        "lower_ci": None,
        "upper_ci": None,
    }


class dummy1Fold:
    def __init__(self) -> None:
        pass

    def split(self, X, y=None, groups=None):
        indices = np.arange(X.shape[0])
        yield indices, indices

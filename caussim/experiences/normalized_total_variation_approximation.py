import json
import logging
from pathlib import Path
from sklearn.base import clone
from typing import Dict

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from caussim.config import DIR2EXPES, SCRIPT_NAME
from caussim.data.causal_df import CausalDf
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV

from caussim.pdistances.mmd import normalized_total_variation
from caussim.score.scores import get_treatment_metrics

logger = logging.getLogger(__name__)


def approximate_n_tv(
    causal_df: CausalDf,
    estimator: Pipeline,
    param_distributions: Dict,
    n_jobs=-1,
    n_iter=10,
    rs_randomized_search=None,
    save: bool = False,
) -> pd.DataFrame:
    """
    Estimate the normalized Total Variation of a causal dataset using a randomized search.
    """
    dataset_description = causal_df.describe()
    a_mean = dataset_description["a_mean"]
    logger.info(
        f"\n🚀 Approximating the normalized total variation of {causal_df.dataset_name} with {estimator}"
    )
    model_name = estimator.steps[-1][0]
    if save:
        h = hash(str(estimator) + json.dumps(param_distributions))
        dir2save = DIR2EXPES / SCRIPT_NAME
        dir2save.mkdir(exist_ok=True, parents=True)
        path2save = (
            dir2save
            / f"n_tv__dataset={causal_df.dataset_name}_overlap={causal_df.overlap_parameter:.4f}_rs={causal_df.random_state}_dgp={causal_df.dgp}_estimator={model_name}_h{h}.csv"
        )
    # first HP search
    if param_distributions is not None:
        logger.info("Fit randomized search for best hps")
        estimator_rsearch = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            scoring="neg_brier_score",
            n_jobs=n_jobs,
            n_iter=n_iter,
            random_state=rs_randomized_search,
        )
        search = estimator_rsearch.fit(causal_df.get_X(), causal_df.get_a())

        best_estimator_ = clone(search.best_estimator_)
        best_params_ = search.best_params_
    else:
        best_estimator_ = clone(estimator)
        best_params_ = {}
    logger.info("Fit classifier")
    y_proba_wo_calibration = cross_val_predict(
        estimator=best_estimator_,
        X=causal_df.get_X(),
        y=causal_df.get_a(),
        n_jobs=n_jobs,
        method="predict_proba",
    )[:, 1]
    nTV_wo_calibration = normalized_total_variation(
        propensity_scores=y_proba_wo_calibration,
        treatment_probability=a_mean,
    )
    scores_wo_calibration = get_treatment_metrics(
        causal_df.get_a(), y_proba_wo_calibration
    )
    scores_wo_calibration = {
        f"{k}_wo_calibration": v for k, v in scores_wo_calibration.items()
    }
    logger.info("Fit calibrated classifier")
    calibrated_estimator_ = CalibratedClassifierCV(
        base_estimator=best_estimator_, n_jobs=n_jobs
    )
    calibrated_estimator_.fit(causal_df.get_X(), causal_df.get_a())
    y_proba_calibrated = calibrated_estimator_.predict_proba(causal_df.get_X())[:, 1]
    nTV_calibrated = normalized_total_variation(
        propensity_scores=y_proba_calibrated,
        treatment_probability=a_mean,
    )
    scores_calibrated = get_treatment_metrics(causal_df.get_a(), y_proba_calibrated)
    scores_calibrated = {f"{k}_calibrated": v for k, v in scores_calibrated.items()}
    logs = pd.DataFrame.from_dict(
        {
            **dataset_description,
            "model_name": model_name,
            **best_params_,
            "oracle_n_tv": dataset_description["d_normalized_tv"],
            "n_tv_wo_calibration": nTV_wo_calibration,
            **scores_wo_calibration,
            "n_tv_calibrated": nTV_calibrated,
            **scores_calibrated,
        },
        orient="index",
    ).transpose()
    if save:
        logger.info(f"Logging results to {path2save}")
        logs.to_csv(path2save, index=False)
    return logs

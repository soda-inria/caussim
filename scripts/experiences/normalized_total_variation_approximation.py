"""
Experience: 
 - Goal: Determine if we can recover good approximation of nTV from learned estimates of the ps. 
 - Datasets: 
   - simulation, overlap in [0.1..2.5], n_reps = 200
   - twins: alpha in [0.1..3], n_reps = 200
   - acic2016: 77 setups, random seed = 1..5
 - Outputs: Estimate the propensity score and the approximation of the nTV using a calibrated classifier, log the oracle nTV from true PS, log also the overlap parameter (simus: overlap parameter, acic2016: overlap setup, twins: sigmoid alpha). 
 - Details: Use logistic regression and histGB as estimators. 
"""

import argparse
from datetime import datetime
from typing import Dict
from copy import deepcopy
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from caussim.data.loading import AVAILABLE_DATASETS, load_dataset, make_dataset_config
from caussim.experiences.normalized_total_variation_approximation import (
    approximate_n_tv,
)

from caussim.experiences.pipelines import HP_KWARGS_HGB, HP_KWARGS_LR
from caussim.experiences.utils import compute_w_slurm


RANDOM_STATE = 0
generator = check_random_state(RANDOM_STATE)
dataset_grids = {
    # "caussim": {"overlap": generator.uniform(0.1, 2.5, size=200), "random_state": [2, 3, 4, 5]},
    # "twins": {"overlap": generator.uniform(0.1, 3, size=200), "random_state": [2, 3, 4, 5]},
    #  "acic_2016": {
    #      "overlap": list(np.arange(1, 78)),
    #      "random_state": list(np.arange(1, 6)),
    # },
}

CLASSIFIERS = {
    "hist_gradient_boosting_classifier": make_pipeline(
        HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    ),
    "logistic_regression": make_pipeline(
        SimpleImputer(), StandardScaler(), LogisticRegression(random_state=RANDOM_STATE)
    ),
}


if __name__ == "__main__":
    t0 = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument("--slurm", dest="slurm", default=False, action="store_true")
    config, _ = parser.parse_known_args()
    config = vars(config)

    simu_grid = []
    for dataset_name, dataset_grid in dataset_grids.items():
        for ds_params in ParameterGrid(dataset_grid):
            dataset_config = make_dataset_config(
                dataset_name,
                **ds_params
            )
            # loading the dataset
            if dataset_name != "acic_2016":
                _, causal_df = load_dataset(dataset_config)
            else:
                from caussim.data.loading import load_acic_2016

                causal_df = load_acic_2016(
                        dgp=int(dataset_config["dgp"]),
                        seed=int(dataset_config["seed"]))
            # iterate over models
            for (classifier_name, estimator), param_distributions in zip(
                CLASSIFIERS.items(), [HP_KWARGS_HGB, HP_KWARGS_LR]
            ):
                run_params = {
                    "causal_df": causal_df,
                    "estimator": estimator,
                    "param_distributions": param_distributions,
                    "rs_randomized_search": RANDOM_STATE,
                    "save": True,
                }
                if config["slurm"]:
                    simu_grid.append(run_params)
                else:
                    approximate_n_tv(**run_params)
    if config["slurm"]:
        compute_w_slurm(approximate_n_tv, simu_grid)

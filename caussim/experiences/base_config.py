"Base configurations for the experiments"
from copy import deepcopy
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    StackingClassifier,
    StackingRegressor,
)

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem

import numpy as np
from sklearn.utils import check_random_state
from caussim.config import DIR2SEMI_SIMULATED_DATA

from caussim.estimation.estimators import (
    SFTLEARNER_LABEL,
    SLEARNER_LABEL,
    TLEARNER_LABEL,
    IdentityTransformer,
)

# TODO: create python objects with pydantic instead of dict: https://gitlab.com/has-sante/public/alpha/scope-sante-data/-/blob/develop/scope_sante_data/scope_sante/models.py

DEFAULT_BASELINE_LINK = {
            "type": "nystroem",
            "params": {
                "n_components": 2,
                "gamma": 0.1,
                "random_state_feat": 2,
    },
}
DEFAULT_SIMU_CONFIG = {
    "n_samples_init": 1000,
    "dim": 2,
    "baseline_link": DEFAULT_BASELINE_LINK,
    "effect_link": deepcopy(DEFAULT_BASELINE_LINK),
    "treatment_link": {
        "type": "linear",
        "params": {},
    },  # Not used because we are using a treatment assignment of type "joint"
    "treatment_assignment": {
        "type": "joint",
        "params": {"sigma": [2, 5], "overlap": 1},
    },
    "effect_size": 0.2,
    "outcome_noise": 0.01,  # 0.01,
    "treatment_ratio": 0.1,
    "random_seed": 0,
    "dataset_name": "caussim",
    "train_size": 2500,
    "test_size": 2500,
    "clip": 0
}


baseline_link_1D = {
            "type": "nystroem",
            "params": {
                "n_components": 10,
                "gamma": 0.1,
                "random_state_feat": 2,
            },
        }
CAUSSIM_1D_CONFIG =  {
            "n_samples_init": 1000,
            "dim": 1,
            "baseline_link": baseline_link_1D,
            "effect_link": deepcopy(baseline_link_1D),
            "treatment_link": {
                "type": "linear",
                "params": {},
            },  # Not used because we are using a treatment assignment of type "joint"
            "treatment_assignment": {
                "type": "joint",
                "params": {"sigma": [2], "overlap": 0.5},
            },
            "effect_size": 2,
            "outcome_noise": 0.001,  # 0.01,
            "treatment_ratio": 0.5,
            "random_seed": 0,
            "dataset_name": "caussim",
            "train_size": 200,
            "test_size": 200,
            "train_seed": 0,
            "test_seed": 1,
            "clip": 0
        }

CATE_CONFIG_ENSEMBLE_NUISANCES = {
    "final_estimator": Ridge(),
    "y_estimator": StackingRegressor(
        estimators=[
            ("histgradientboostingregressor", HistGradientBoostingRegressor()),
            (
                "featurized_ridge",
                Pipeline(
                    [
                        ("featurizer", IdentityTransformer()),
                        ("scaler", StandardScaler()),
                        ("ridge", Ridge()),
                    ]
                ),
            ),
        ]
    ),
    "y_hyperparameters": {
        "featurized_ridge__ridge__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
        "featurized_ridge__featurizer__random_state": [0, 1, 2, 3, 4, 5],
        "histgradientboostingregressor__learning_rate": [0.01, 0.1, 1],
        "histgradientboostingregressor__max_leaf_nodes": [10, 20, 30, 50],
    },
    "y_scoring": "r2",
    "a_estimator": StackingClassifier(
        estimators=[
            ("histgradientboostingclassifier", HistGradientBoostingClassifier()),
            (
                "featurized_logisticregression",
                Pipeline(
                    [
                        ("featurizer", IdentityTransformer()),
                        ("scaler", StandardScaler()),
                        ("logisticregression", LogisticRegression()),
                    ]
                ),
            ),
        ]
    ),
    "a_hyperparameters": {
        "featurized_logisticregression__logisticregression__C": [
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1,
            10,
            100,
        ],
        "featurized_logisticregression__featurizer__random_state": [0, 1, 2, 3, 4, 5],
        "histgradientboostingclassifier__learning_rate": [
            0.01,
            0.1,
            1,
        ],
        "histgradientboostingclassifier__max_leaf_nodes": [10, 20, 30, 50],
    },
    "a_scoring": "roc_auc",
    "n_iter_random_search": 20,
    "n_splits": 1,
    "rs_hp_search": 0,
    "rs_kfold": 0,
    "train_ratio": 0.5,
    "rs_train_split": 0,
    "separate_train_set_ratio": 0,
    "rs_separate_train_set": 2
}


CATE_CONFIG_LOGISTIC_NUISANCES = {
    "final_estimator": Ridge(),
    "y_estimator": Pipeline(
        [
            ("featurizer", IdentityTransformer()),
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]
    ),
    "y_hyperparameters": {
        "ridge__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
        "featurizer__random_state": [0, 1, 2, 3, 4, 5],
    },
    "y_scoring": "r2",
    "a_estimator": Pipeline(
        [
            ("featurizer", IdentityTransformer()),
            ("scaler", StandardScaler()),
            ("logisticregression", LogisticRegression()),
        ]
    ),
    "a_hyperparameters": {
        "logisticregression__C": [
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1,
            10,
            100,
        ],
        "featurizer__random_state": [0, 1, 2, 3, 4, 5],
    },
    "a_scoring": "roc_auc",
    "n_iter_random_search": 20,
    "n_splits": 1,
    "rs_hp_search": 0,
    "rs_kfold": 0,
    "train_ratio": 0.5,
    "rs_train_split": 0,
    "separate_train_set_ratio": 0,
    "rs_separate_train_set": 2
}


CANDIDATE_FAMILY_HGB = [
    {
        "meta_learner_name": SLEARNER_LABEL,
        "estimator": "hist_gradient_boosting",
        "hp_kwargs": {
            "final_estimator__learning_rate": [0.01, 0.1, 1],
            "final_estimator__max_leaf_nodes": [25, 27, 30, 32, 35, 40],
            # "final_estimator__random_state": [0],
        },
    }
]


CANDIDATE_FAMILY_HGB_UNDERFIT = [
    {
        "meta_learner_name": SLEARNER_LABEL,
        "estimator": "hist_gradient_boosting",
        "hp_kwargs": {
            "final_estimator__learning_rate": [0.01, 0.1, 1],
            "final_estimator__max_leaf_nodes": [25, 30, 35, 40],
            "final_estimator__early_stopping": [True],
            "final_estimator__loss": ["absolute_error", "squared_error"]
        },
    }
]

CANDIDATE_FAMILY_RIDGE_TLEARNERS = [
    {
        "meta_learner_name": TLEARNER_LABEL,
        "estimator": "ridge",
        "featurizer": Nystroem(
                            random_state=0,
                            n_components=2,
                            gamma=1.0,
                        ),
        "hp_kwargs": {
            "final_estimator__alpha": [1e-3, 0.01, 0.1, 1, 10, 100],
            "featurizer__random_state": np.arange(10).tolist(),
        },
    },
    {
        "meta_learner_name": SFTLEARNER_LABEL,
        "estimator": "ridge",
        "featurizer": Nystroem(
                            random_state=0,
                            n_components=2,
                            gamma=1.0,
                        ),
        "hp_kwargs": {
            "final_estimator__alpha": [1e-3, 0.01, 0.1, 1, 10, 100],
            "featurizer__random_state": np.arange(10).tolist(),
        },
    },
]


CANDIDATE_FAMILY_RIDGE_WO_FEAT = [
    {
        "meta_learner_name": TLEARNER_LABEL,
        "estimator": "ridge",
        "featurizer": IdentityTransformer(),
        "hp_kwargs": {
            "final_estimator__alpha": [1e-3, 0.01, 0.1, 1, 10, 100],
        },
    },
    {
        "meta_learner_name": SLEARNER_LABEL,
        "estimator": "ridge",
        "featurizer": IdentityTransformer(),
        "hp_kwargs": {
            "final_estimator__alpha": [1e-3, 0.01, 0.1, 1, 10, 100],
        },
    },
]

CANDIDATE_FAMILY_RIDGE_POLY3 = [
    {
        "meta_learner_name": SLEARNER_LABEL,
        "estimator": "poly3",
        "hp_kwargs": {
            "final_estimator__ridge__alpha": [1e-3, 0.1, 1, 10],
        },
    },
    
]

CANDIDATE_FAMILY_RF = [
    {
        "meta_learner_name": SLEARNER_LABEL,
        "estimator": "random_forest",
        "hp_kwargs": {
            "final_estimator__n_estimators": [4],
            "final_estimator__max_depth": [3, 4, 5]
        },
    },
    {
        "meta_learner_name": TLEARNER_LABEL,
        "estimator": "random_forest",
        "hp_kwargs": {
            "final_estimator__n_estimators": [4],
            "final_estimator__max_depth": [3, 4, 5]
        },
    }
]

CANDIDATE_FAMILY_MIXED = [
    {
        "meta_learner_name": SLEARNER_LABEL,
        "estimator": "poly3",
        "hp_kwargs": {
            "final_estimator__ridge__alpha": [0.01, 1],
            "final_estimator__polynomialfeatures__degree": [3, 4, 5]
        }
    },
    {
        "meta_learner_name": TLEARNER_LABEL,
        "estimator": "poly3",
        "hp_kwargs": {
            "final_estimator__ridge__alpha": [0.01, 1],
            "final_estimator__polynomialfeatures__degree": [3, 4, 5]
        }
    },
    {
        "meta_learner_name": SLEARNER_LABEL,
        "estimator": "random_forest",
        "hp_kwargs": {
            "final_estimator__n_estimators": [2, 4],
            "final_estimator__max_depth": [2, 4, 5]
        },
    },
    {
        "meta_learner_name": TLEARNER_LABEL,
        "estimator": "random_forest",
        "hp_kwargs": {
            "final_estimator__n_estimators": [2, 4],
            "final_estimator__max_depth": [2, 4, 5]
        },
    },
    {
        "meta_learner_name": SLEARNER_LABEL,
        "estimator": "spline3",
        "hp_kwargs": {
            "final_estimator__splinetransformer__n_knots": [2, 5, 10],
            "final_estimator__ridge__alpha": [0.01, 1]
        },
    },
    {
        "meta_learner_name": TLEARNER_LABEL,
        "estimator": "spline3",
        "hp_kwargs": {
            "final_estimator__splinetransformer__n_knots": [2, 5, 10],
            "final_estimator__ridge__alpha": [0.01, 1]
        },
    }
]

CANDIDATE_FAMILY_RIDGE_SLEARNER = [
    {
        "meta_learner_name": SLEARNER_LABEL,
        "estimator": "ridge",
        "hp_kwargs": {
            "final_estimator__alpha": [1e-3, 0.01, 0.1, 1, 10, 100],
            "featurizer__random_state": np.arange(10).tolist(),
        },
    }
]

CANDIDATE_FAMILY_ATE_HETEROGENEITY = [
    {
        "meta_learner_name": SLEARNER_LABEL,
        "estimator": "hist_gradient_boosting",
        "hp_kwargs": {
            "final_estimator__learning_rate": [0.01, 0.1],
        },
    },
    {
        "meta_learner_name": SLEARNER_LABEL,
        "estimator": "ridge",
        "hp_kwargs": {
            "final_estimator__alpha": [1e-3, 0.1],
        },
    },
    {
        "meta_learner_name": TLEARNER_LABEL,
        "estimator": "ridge",
        "hp_kwargs": {
            "final_estimator__alpha": [1e-3, 0.1],
        },
    },
]

# DATASET_GRIDS
RANDOM_STATE = 0
generator = check_random_state(RANDOM_STATE)

ACIC_2018_PARAMS = pd.read_csv(DIR2SEMI_SIMULATED_DATA / "acic_2018" / "scaling" / "params.csv")

DATASET_GRID_FULL_EXPES = [
    {"dataset_name": ["twins"],"overlap": generator.uniform(0.1, 3, size=100), "random_state": list(np.arange(10))},
    {"dataset_name": ["acic_2018"], "ufid": ACIC_2018_PARAMS.loc[ACIC_2018_PARAMS["size"] <=5000, "ufid"].values},
    {"dataset_name": ["acic_2016"], "overlap": list(range(1, 78)),"random_state": list(range(1, 11))},
    {"dataset_name": ["caussim"], "overlap": generator.uniform(0, 2.5, size=100), "random_state":list(range(1, 11))} 
]

DATASET_GRID_TWO_THREE_SETS = [
    {"dataset_name": ["twins"],"overlap": generator.uniform(0.1, 3, size=100), "random_state": list(np.arange(10))},
    {"dataset_name": ["acic_2018"], "ufid": ACIC_2018_PARAMS.loc[ACIC_2018_PARAMS["size"] <=5000, "ufid"].values},
    {"dataset_name": ["acic_2016"], "overlap": list(range(1, 78)),"random_state": list(range(1, 11))},
    {"dataset_name": ["caussim"], "overlap": generator.uniform(0, 2.5, size=100), "random_state":list(range(1, 11))} 
]

DATASET_GRID_EXTRAPOLATION_RESIDUALS = [
    {"dataset_name": ["acic_2016"], "overlap": [2],"random_state": [9]}, # bad overlap 0.63, good R-risk R_risk IS2
    {"dataset_name": ["acic_2016"], "overlap": [76],"random_state": [6]}, # good overlap 0.128 good R-risk and R_risk IS2
    {"dataset_name": ["acic_2016"], "overlap": [11],"random_state": [4]}, # bad overlap 0.47, bad R-risk and R risk IS2
    {"dataset_name": ["acic_2016"], "overlap": [77],"random_state": [4]}, # good overlap 0.14, bad R-risk and R risk IS2
    {"dataset_name": ["acic_2016"], "overlap": [30],"random_state": [1]},  # setup overlap full
    {"dataset_name": ["caussim"], "overlap":[0.8], "random_state": [2]}, 
    #{"dataset_name": ["caussim"], "overlap":[1.87197064], "random_state": [2]}, # bad overlap 0.84, bad R_risk_IS2, R risk ok  -> why I don't have the oracle tau risk IS2 (printing)
    {"dataset_name": ["caussim"], "overlap":[0.5], "random_state": [52]}, 
    {"dataset_name": ["caussim"], "overlap":[1.5], "random_state": [10]}, 
]
"""
Score several g-formula models with mu_iptw_risk (reweighted mse) and mu_risk (mse on y) on different semi-simulated datasets
"""
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from sklearn.utils import check_random_state
from tqdm import tqdm
from copy import deepcopy
from caussim.data.loading import make_dataset_config
from caussim.experiences.causal_scores_evaluation import run_causal_scores_eval
from caussim.experiences.pipelines import *


from caussim.experiences.base_config import (
    CANDIDATE_FAMILY_HGB_UNDERFIT,
    CANDIDATE_FAMILY_RIDGE_TLEARNERS,
    CATE_CONFIG_ENSEMBLE_NUISANCES,
    CANDIDATE_FAMILY_HGB,
    CATE_CONFIG_LOGISTIC_NUISANCE,
    DATASET_GRID_EXTRAPOLATION_RESIDUALS,
    DATASET_GRID_FULL_EXPES,
)

from caussim.experiences.utils import compute_w_slurm, set_causal_score_xp_name

RANDOM_STATE = 0
generator = check_random_state(RANDOM_STATE)

# DATASET_GRID = [
#     {"dataset_name": ["acic_2016"], "overlap": list(range(5)),"random_state": list(range(1,2))},
#     {"dataset_name": ["acic_2016"], "overlap": list(range(20)),"random_state": list(range(1,2))},
#     {"dataset_name": ["acic_2016"], "overlap": list(range(30)),"random_state": list(range(1,2))} # overlap full
#     ]

DATASET_GRID = [
    # {"dataset_name": ["acic_2016"], "dgp": list(range(1, 78)),"random_state": list(range(1, 11))}
    {
        "dataset_name": ["caussim"],
        "overlap": generator.uniform(0, 2.5, size=100),
        "random_state": list(range(1, 4)),
        "treatment_ratio": [0.25, 0.5, 0.75],
    }
    # {"dataset_name": ["acic_2018"], "ufid": ACIC_2018_PARAMS.loc[ACIC_2018_PARAMS["size"] == 5000, "ufid"].values},
    # {"dataset_name": ["acic_2016"], "dgp": list(range(1, 78)),"random_state": list(range(1, 6))},
    # {"dataset_name": ["caussim"], "overlap": generator.uniform(0, 2.5, size=100), "random_state":list(range(1, 11))}
    # {"dataset_name": ["twins"],"overlap": generator.uniform(0.1, 3, size=100), "random_state": list(np.arange(5))},
    ## Add a configuration for nuisance, train, test sets.
    # {
    #     "dataset_name": ["caussim"],
    #     "overlap": generator.uniform(0, 2.5, size=100),
    #     "random_state": list(range(1, 4)),
    #     "treatment_ratio": [0.25, 0.5, 0.75],
    #     "nuisance_set_size": [2500],
    # }
]
# DATASET_GRID = DATASET_GRID_FULL_EXPES


# ### Evaluate several dgps ### #
if __name__ == "__main__":
    t0 = datetime.now()
    parser = argparse.ArgumentParser()
    # parser.add_argument("--xp_name", type=str,default=None,help="xp folder to consolidate",)
    parser.add_argument("--slurm", dest="slurm", default=False, action="store_true")
    parser.add_argument(
        "--extrapolation_plot",
        dest="extrapolation_plot",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--write_to_parquet",
        dest="write_to_parquet",
        default=False,
        action="store_true",
    )
    config, _ = parser.parse_known_args()
    config = vars(config)

    expe_timestamp = datetime.now()
    # Set important parameters for loop
    cate_config = deepcopy(CATE_CONFIG_ENSEMBLE_NUISANCES)
    cate_config["test_ratio"] = 0.5
    cate_config["rs_test_split"] = 0
    # Loop on simulations
    # xp_name = config['xp_name']
    simu_grid = []
    for dataset_grid in DATASET_GRID:
        dataset_name = dataset_grid["dataset_name"][0]
        if dataset_name == "caussim":
            candidate_estimators_grid = deepcopy(CANDIDATE_FAMILY_RIDGE_TLEARNERS)
        else:
            candidate_estimators_grid = deepcopy(CANDIDATE_FAMILY_HGB)
        xp_name = set_causal_score_xp_name(
            dataset_name=dataset_name,
            dataset_grid=dataset_grid,
            cate_config=cate_config,
            candidate_estimators_grid=candidate_estimators_grid,
        )
        for dataset_setup in tqdm(ParameterGrid(dataset_grid)):
            dataset_config = make_dataset_config(**dataset_setup)
            cate_config = deepcopy(cate_config)
            candidate_estimators_grid = deepcopy(candidate_estimators_grid)
            if config["slurm"]:
                simu_grid.append(
                    {
                        "dataset_config": dataset_config,
                        "cate_config": cate_config,
                        "candidate_estimators_grid": candidate_estimators_grid,
                        "xp_name": xp_name,
                        "extrapolation_plot": config["extrapolation_plot"],
                        "write_to_parquet": config["write_to_parquet"],
                    }
                )
            else:
                run_causal_scores_eval(
                    dataset_config=dataset_config,
                    cate_config=cate_config,
                    candidate_estimators_grid=candidate_estimators_grid,
                    xp_name=xp_name,
                    extrapolation_plot=config["extrapolation_plot"],
                    write_to_parquet=config["write_to_parquet"],
                )
    if config["slurm"]:
        compute_w_slurm(run_causal_scores_eval, simu_grid)
    print(f"\n##### Cycle of simulations ends ##### \n Duration: {datetime.now() - t0}")

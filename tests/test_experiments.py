from copy import deepcopy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from caussim.data.loading import make_dataset_config
from caussim.estimation.estimators import TLEARNER_LABEL
from caussim.experiences.base_config import  CATE_CONFIG_LOGISTIC_NUISANCE
from caussim.experiences.normalized_total_variation_approximation import (
    approximate_n_tv,
)
from caussim.experiences.pipelines import HP_KWARGS_LR
import pytest

from scripts.experiences.causal_scores_evaluation import run_causal_scores_eval

def test_approximate_ntv(twins_dataset):
    clf = make_pipeline(LogisticRegression())
    logs = approximate_n_tv(
        causal_df=twins_dataset,
        estimator=clf,
        param_distributions=HP_KWARGS_LR,
        save=True,
    )
    oracle_ntv = logs["oracle_n_tv"]
    # diff_wo_calibration = np.abs(oracle_ntv - logs["n_tv_wo_calibration"])
    # diff_calibrated = np.abs(oracle_ntv - logs["n_tv_calibrated"])

    assert logs["bs_wo_calibration"].values[0] >= logs["bs_calibrated"].values[0]
    # assert diff_wo_calibration >= diff_calibrated # does not work : so calibration is not sufficient for this ? Need GL optimization ? 

#TODO: run profiling here to know what to accelerate
test_dataset_grid = {
    "twins": {"overlap": 0.5, "random_state":1},
    "acic_2018": {"ufid": "a957b431a74a43a0bb7cc52e1c84c8ad"},
    "acic_2016": {"dgp": 1, "random_state": 1},
    "caussim": {"overlap": 1, "random_state": 1},
}
test_candidate_family = [
    {
        "meta_learner_name": TLEARNER_LABEL,
        "estimator": "ridge",
        "hp_kwargs": {
            "final_estimator__alpha": [0.01, 0.1],
            "featurizer__random_state": [0, 1],
        },
    },
]
@pytest.mark.parametrize(
    "dataset_name", 
    #["caussim", "twins", 'acic_2018', 'acic_2016']
    ["caussim"]
    )   
def test_causal_scores_evaluation(dataset_name):
    dataset_config = make_dataset_config(
                dataset_name=dataset_name,
                **test_dataset_grid[dataset_name]
            )
    cate_config = deepcopy(CATE_CONFIG_LOGISTIC_NUISANCE)
    cate_config["test_ratio"] = 0.5
    cate_config["rs_test_split"] = 0
    candidate_estimators_grid = deepcopy(test_candidate_family)
    
    xp_results = run_causal_scores_eval(
        dataset_config=dataset_config,
        cate_config=cate_config,
        candidate_estimators_grid=candidate_estimators_grid,
        xp_name="test",
        extrapolation_plot=True
    ).reset_index(drop=True).to_dict(orient="index")[0]
    #TODO: better tests ? 
    for k, v in dataset_config.items():
        assert xp_results[k] == v
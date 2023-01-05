"""
Score several g-formula models with mu_iptw_risk (reweighted mse) and mu_risk (mse on y) on the [ACIC 2016 dataset](https://github.com/vdorie/aciccomp/tree/master/2016)
"""
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    train_test_split,
    ParameterGrid,
)
from copy import deepcopy

from tqdm import tqdm
import yaml
from caussim.config import DIR2EXPES, SCRIPT_NAME
from caussim.data.causal_df import CausalDf
from caussim.data.loading import load_acic_2016
from caussim.estimation.estimators import (
    CateEstimator,
    get_treatment_and_covariates,
    set_meta_learner,
)

from caussim.experiences.pipelines import *
from caussim.estimation.estimation import get_selection_metrics
import pyarrow as pa
import pyarrow.parquet as pq
from caussim.experiences.base_config import (
    CATE_CONFIG_ENSEMBLE_NUISANCES,
    CANDIDATE_FAMILY_ATE_HETEROGENEITY,
)

from caussim.experiences.utils import compute_w_slurm

SIMULATION_DGPS = range(1, 76)  # [2, 10, 18, 35, 54, 69]  # up to 1-77
# SIMULATION_SEEDS = range(1, 2)  # range(1, 11)  # up to 101
TEST_SPLIT_SEEDS = range(0, 10)

DATASET_CONFIG_ACIC_16 = {
    "dataset_name": "acic_2016",
    "simulation_param": 1,
    "simulation_seed": 1,
    "trim": None,
    "write_data": False,
}


# %%
def run_acic_2016_heterogeneity(simu_config, learner_grid, xp_name=None):
    expe_timestamp = datetime.now()
    if xp_name is None:
        xp_name = f"{expe_timestamp.strftime('%Y-%m-%d-%H-%M-%S')}_{SCRIPT_NAME}"
    path2results = DIR2EXPES / SCRIPT_NAME / xp_name
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        filename=f"{DIR2EXPES}/acic_2016_scoring.log",
    )
    logger = logging.getLogger("__name__")
    learning_family = []
    for learner_estimator_params in learner_grid:
        learning_family += ParameterGrid(learner_estimator_params["hp_kwargs"])
    nb_runs = len(learning_family)
    logger.info(
        f"""\n
    -----NEW EXPERIMENT-----
    Number of simulations that will be run: {nb_runs}
    Learner grid: {learner_grid}
    """
    )
    # load a dgp
    dgp_sample = load_acic_2016(
        dgp=simu_config["simulation_param"],
        seed=simu_config["simulation_seed"],
        trim=simu_config["trim"],
    )

    df_train, df_test = train_test_split(
        dgp_sample.df,
        test_size=simu_config["test_ratio"],
        random_state=simu_config["rs_test_split"],
    )

    df_train = CausalDf(df_train.reset_index(drop=True))
    df_test = CausalDf(df_test.reset_index(drop=True))
    X_test, y_test = df_test.get_aX_y()
    X_train, y_train = df_train.get_aX_y()
    # TODO: we might want to keep e nuisance learning to derive IPW estimations.
    # base_meta_learner = set_meta_learner(
    #     SLEARNER_LABEL, final_estimator=cate_config["final_estimator"]
    # )
    #
    # cate_estimator = CateEstimator(
    #     meta_learner=base_meta_learner,
    #     a_estimator=cate_config["a_estimator"],
    #     y_estimator=cate_config["y_estimator"],
    #     a_hyperparameters=cate_config["a_hyperparameters"],
    #     y_hyperparameters=cate_config["y_hyperparameters"],
    #     a_scoring=cate_config["a_scoring"],
    #     y_scoring=cate_config["y_scoring"],
    #     n_iter=cate_config["n_iter_random_search"],
    #     cv=cate_config["n_splits"],
    #     random_state_hp_search=cate_config["rs_hp_search"],
    # )

    path2results.mkdir(exist_ok=True, parents=True)
    with open(path2results / "simu.yaml", "w", encoding="utf8") as f:
        yaml.dump({**simu_config, **{"learner_grid": learner_grid}}, f, sort_keys=False)

    # ### Prefit nuisance models ### #
    # cate_estimator.fit_nuisances(X=X_train, y=y_train)

    # ### Iterate over candidate estimators ### #
    for candidate_model_params in learner_grid:
        # iterate over meta-learner parameters
        for parameter_grid in list(ParameterGrid(candidate_model_params["hp_kwargs"])):
            t0 = datetime.now()

            candidate_meta_learner = set_meta_learner(
                meta_learner_name=candidate_model_params["meta_learner_name"],
                final_estimator=REGRESSORS[candidate_model_params["estimator"]],
            )
            candidate_meta_learner.set_params(**parameter_grid)

            candidate_meta_learner.fit(X_train, y_train)
            # We are doing cross validation for score predictions (ie reusing fitted model nuisances and cv associated)
            a_test, X_cov_test = get_treatment_and_covariates(X_test)
            test_predictions = {
                "hat_tau": candidate_meta_learner.predict_cate(X=X_test),
                "hat_mu_1": candidate_meta_learner.predict_cate(
                    np.column_stack((np.ones_like(a_test), X_cov_test))
                ),
                "hat_mu_0": candidate_meta_learner.predict_cate(
                    np.column_stack((np.zeros_like(a_test), X_cov_test))
                ),
                "check_m": np.ones_like(y_test) * y_test.mean(),
                "check_e": np.ones_like(a_test) * a_test.mean(),
            }
            metrics = get_selection_metrics(
                causal_df=df_test, predictions=test_predictions
            )
            df_test_info = df_test.describe(prefix="test_")
            df_dgp_info = dgp_sample.describe(prefix="dgp_")

            # log_cate_config = deepcopy(cate_config)
            # log_cate_config["y_estimator"] = str(log_cate_config["y_estimator"])
            # log_cate_config["a_estimator"] = str(log_cate_config["a_estimator"])
            # logging results
            xp_param = {
                **simu_config,
                #    **log_cate_config,
                **parameter_grid,
                **df_test_info,
                **df_dgp_info,
                **metrics,
                **{"cate_candidate_model": candidate_model_params["estimator"]},
                **{"meta_learner_name": candidate_model_params["meta_learner_name"]},
            }
            xp_param["compute_time"] = datetime.now() - t0
            run_id = hash(
                t0.strftime("%Y-%m-%d-%H-%M-%S")
                + json.dumps(parameter_grid)
                + json.dumps(metrics)
            )
            log = pd.DataFrame.from_dict(xp_param, orient="index").transpose()
            log.to_csv(path2results / (f"run_{run_id}.csv"), index=False)
            if simu_config["write_data"]:
                test_data_to_write = pd.concat([df_test.df, test_predictions], axis=1)
                test_data_to_write["run_id"] = run_id
                table = pa.Table.from_pandas(test_data_to_write)
                pq.write_to_dataset(
                    table, root_path=str(path2results / "test_data.parquet")
                )

            total_compute_time = datetime.now() - expe_timestamp
            end_signal = f"""
                Terminate experiment {xp_name} in {total_compute_time}, \n see manual logging at {path2results}
                -----EXPERIMENT ENDS-----\n
                """
            print(end_signal)
            logger.info(end_signal)
    return xp_name


# ### Evaluate several dgps ### #
if __name__ == "__main__":
    t0 = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xp_name",
        type=str,
        default=None,
        help="xp folder to consolidate",
    )
    parser.add_argument("--slurm", dest="slurm", default=False, action="store_true")

    config, _ = parser.parse_known_args()
    config = vars(config)
    xp_name = config["xp_name"]
    if xp_name is None:
        expe_timestamp = datetime.now()
        xp_name = f"{expe_timestamp.strftime('%Y-%m-%d-%H-%M-%S')}_{SCRIPT_NAME}"
        print("New Experiment at {}".format(xp_name))
    else:
        xp_name = Path(xp_name).stem
        print("Consolidating Experiment at {}".format(xp_name))

    # Set important parameters
    simu_config = deepcopy(DATASET_CONFIG_ACIC_16)
    # cate_config = deepcopy(BASE_CATE_CONFIG)
    simu_config["test_ratio"] = 0.5
    simu_config["rs_test_split"] = 0
    learner_grid = deepcopy(CANDIDATE_FAMILY_ATE_HETEROGENEITY)
    # loop config
    test_split_seeds = deepcopy(TEST_SPLIT_SEEDS)
    simulation_dgps = deepcopy(SIMULATION_DGPS)

    # Loop on simulations
    simu_grid = []

    for seed in tqdm(test_split_seeds, desc="outer", position=0):
        for dgp in tqdm(simulation_dgps, desc="inner loop", position=1, leave=False):
            simu_config = deepcopy(simu_config)
            learner_grid = deepcopy(learner_grid)
            simu_config["rs_test_split"] = seed
            simu_config["simulation_param"] = dgp
            if config["slurm"]:
                simu_grid.append(
                    {
                        "simu_config": simu_config,
                        "learner_grid": learner_grid,
                        "xp_name": xp_name,
                    }
                )
            else:
                run_acic_2016_heterogeneity(
                    simu_config=simu_config,
                    learner_grid=learner_grid,
                    xp_name=xp_name,
                )
    if config["slurm"]:
        compute_w_slurm(run_acic_2016_heterogeneity, simu_grid)
    print(f"\n##### Cycle of simulations ends ##### \n Duration: {datetime.now() - t0}")

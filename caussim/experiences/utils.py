from copyreg import pickle
import logging
from pathlib import Path
import pickle
import re
import time
from typing import Dict, List
import numpy as np

import pandas as pd
import submitit
import yaml
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from caussim.pdistances.mmd import mmd_rbf, total_variation_distance
from caussim.config import DIR2CACHE, DIR2EXPES
from caussim.estimation.estimators import SLEARNER_LABEL, set_meta_learner
from caussim.data.simulations import CausalSimulator, get_transformed_space_from_simu
from joblib import Memory

memory = Memory(DIR2CACHE, verbose=0)


def set_causal_score_xp_name(dataset_name:str, dataset_grid, cate_config, candidate_estimators_grid) -> str:
    _supported_datasets = ["acic_2016", "acic_2018", "twins", "caussim"]
    nuisance_reg = re.search("gradientboosting|randomforest", str(cate_config["y_estimator"]).lower())
    if nuisance_reg is not None:
        estimator_type = "non_linear"
    else:
        estimator_type = "linear"
    candidate_family_type = candidate_estimators_grid[0]["estimator"]
    xp_name = f"{dataset_name}__nuisance_{estimator_type}__candidates_{candidate_family_type}__"
    if dataset_name == "acic_2016":
        xp_name += f"dgp_{min(dataset_grid['dgp'])}-{max(dataset_grid['dgp'])}__rs_{min(dataset_grid['random_state'])}-{max(dataset_grid['random_state'])}"
    elif dataset_name == "acic_2018":
        xp_name += f"first_uid_{len(dataset_grid['ufid'])}"
    elif dataset_name == "twins":
        xp_name += f"overlap_{int(100*min(dataset_grid['overlap'])):02d}-{int(100*max(dataset_grid['overlap'])):02d}__rs_{min(dataset_grid['random_state'])}-{max(dataset_grid['random_state'])}"
    elif dataset_name == "caussim":
        xp_name += f"overlap_{int(100*min(dataset_grid['overlap'])):02d}-{int(100*max(dataset_grid['overlap'])):02d}"
    else:
        raise ValueError(f"Supported datasets are: {_supported_datasets}, got {dataset_name}")
    return xp_name

@memory.cache
def get_candidates_family(
    fixed_simulation: CausalSimulator, outcome_model: BaseEstimator, learner_grid: Dict
):
    """Given a simulation and a list of different parameters for the learner, train a family of estimators.

    Args:
        fixed_simulation (CausalSimulator): Simulation to use as a fixed parameter.
        outcome_model (BaseEstimator): Base outcome model to be evaluated
        learner_grid (Dict): Dictionary of parameters to be tested, support:
            - random state for the learner,
            - random state for the featurization step
            - overlap failure parameter for the sample generated samples from the simulation. Higher parameter yields less overlap thus more heterogeneous populations (and learned response surfaces).

    Returns:
        List[Dict]: List of estimators and their parameters
    """
    candidate_estimators_family = []
    for learner_params in tqdm(list(ParameterGrid(learner_grid))):
        fixed_simulation.params["treatment_assignment"]["overlap"] = learner_params[
            "overlap_learner"
        ]
        candidate_meta_learner = set_meta_learner(
            SLEARNER_LABEL,
            final_estimator=outcome_model,
            featurizer=fixed_simulation.baseline_pipeline.pipeline.named_steps[
                "featurizer"
            ],
        )

        candidate_meta_learner.set_params(
            **{
                "final_estimator__alpha": learner_params["regression__alpha"],
                "final_estimator__random_state": learner_params["rs_learner"],
                "featurizer__random_state": learner_params["rs_learner"],
            }
        )
        # TODO: Forcing SLearner for all, not an evident choice.
        df_train = fixed_simulation.sample(learner_params["num_samples"])
        train_tv = total_variation_distance(df_train["e"], 1 - df_train["e"])
        Z_control, Z_treated = get_transformed_space_from_simu(
            fixed_simulation, df_train
        )
        train_transformed_mmd = mmd_rbf(Z_control, Z_treated)
        train_mmd = mmd_rbf(
            df_train.loc[df_train["a"] == 1, fixed_simulation.x_cols],
            df_train.loc[df_train["a"] == 0, fixed_simulation.x_cols],
        )
        candidate_meta_learner.fit(
            np.vstack((df_train["a"].values, df_train[fixed_simulation.x_cols].values)),
            df_train["y"].values,
        )
        candidate_estimators_family.append(
            {
                "meta_learner": candidate_meta_learner,
                "parameters": {
                    **learner_params,
                    "train_tv": train_tv,
                    "train_mmd": train_mmd,
                    "train_transformed_mmd": train_transformed_mmd,
                },
            }
        )
    return candidate_estimators_family


def consolidate_xps(xp_name: str, xp_save: str = None):
    """Concatenate all xps in a folder into a single dataframe.

    Args:
        xp_name (str): Folder of raw individual xps to consolidate.
        xp_save (str, optional): Folder of consolidated xps to consolidate with new raw xps. Defaults to None.
    """
    xp_path = Path(xp_name)
    xp_type = xp_path.parent.stem
    xp_name = xp_path.stem
    dir2expe = DIR2EXPES / xp_type / xp_name
    if xp_save is None:
        dir2save = DIR2EXPES / f"{xp_type}_save" / xp_name
    else:
        dir2save = Path(xp_save)
    dir2save.mkdir(exist_ok=True, parents=True)

    # Save simulation configuration
    if (dir2expe / "simu.yaml").exists():
        with open(dir2expe / "simu.yaml", "r") as f:
            simu_config = yaml.load(f, yaml.FullLoader)
        with open(dir2save / "simu.yaml", "w") as f:
            yaml.dump(simu_config, f)
    elif (dir2expe / "simu.pkl").exists():
        with open(dir2expe / "simu.pkl", "rb") as f:
            simu_config = pickle.load(f)
        with open(dir2save / "simu.pkl", "wb") as f:
            pickle.dump(simu_config, f)
    else:
        logging.warning("No configuration file found.")
    run_csvs = [f for f in list(dir2expe.iterdir()) if f.suffix == ".csv"]
    print("reading at {}".format(str(dir2expe)))
    run_logs = pd.concat([pd.read_csv(f) for f in run_csvs], axis=0)
    run_logs["run_id"] = [f.stem for f in run_csvs]

    if (dir2save / "run_logs.csv").is_file():
        previous_df = pd.read_csv(dir2save / "run_logs.csv")
        new_df = pd.concat([previous_df, run_logs], axis=0)
        new_df.to_csv(dir2save / "run_logs.csv", index=False)
    else:
        run_logs.to_csv(dir2save / "run_logs.csv", index=False)


def compute_w_slurm(func, list_parameters: List[Dict], n_cpus: int = 40):
    """Wrap a function and a list of parameters to run with parallelism using slurm"""
    #
    executor = get_executor_marg(
        "awesome_task", timeout_hour=10, n_cpus=n_cpus, max_parallel_tasks=8, gpu=False
    )

    # Run the computation on SLURM cluster with `submitit`
    print("Submitting jobs...", end="", flush=True)
    with executor.batch():
        tasks = [executor.submit(func, **parameters) for parameters in list_parameters]

    t_start = time.time()
    print("done")


def get_executor_marg(
    job_name, timeout_hour=60, n_cpus=10, max_parallel_tasks=20, gpu=False
):
    """Return a submitit executor to launch various tasks on a SLURM cluster.

    Parameters
    ----------
    job_name: str
        Name of the tasks that will be run. It will be used to create an output
        directory and display task info in squeue.
    timeout_hour: int
        Maximal number of hours the task will run before being interupted.
    n_cpus: int
        Number of CPUs requested for each task.
    max_parallel_tasks: int
        Maximal number of tasks that will run at once. This can be used to
        limit the total amount of the cluster used by a script. Note: [Margaret has 41 CPU nodes and 8 GPU nodes](https://gitlab.inria.fr/clusters-saclay/margaret/-/wikis/hardware)
    gpu: bool
        If set to True, require one GPU per task.
    """

    executor = submitit.AutoExecutor(job_name)
    executor.update_parameters(
        timeout_min=180,
        slurm_job_name=job_name,
        slurm_time=f"{timeout_hour}:00:00",
        slurm_array_parallelism=max_parallel_tasks,
        slurm_additional_parameters={
            "ntasks": 1,
            "cpus-per-task": n_cpus,
            "distribution": "block:block",
        },
    )
    if gpu:
        executor.update_parameters(
            slurm_gres=f"gpu:1",
            slurm_setup=[
                "#SBATCH -C v100-32g",  # Require a specific resource
                "module purge",
                "module load cuda/10.1.2 "  # Load drivers
                "cudnn/7.6.5.32-cuda-10.1 nccl/2.5.6-2-cuda",
            ],
        )
    return executor

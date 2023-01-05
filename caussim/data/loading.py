from copy import deepcopy
import logging
from typing import Dict, Tuple, Union
from joblib import Memory
import numpy as np

import pandas as pd
from sklearn.utils import check_random_state


from caussim.config import (
    DIR2ACIC_2016,
    DIR2CACHE,
    CAUSAL_DF_COLUMNS,
    DIR2ACIC_2018,
    DIR2ACIC_2018_CF,
    DIR2ACIC_2018_F,
    DIR2TWINS,
    PATH2ACIC_2018_X,
    ROOT_DIR,
)
from caussim.data.causal_df import CausalDf


from caussim.data.simulations import CausalSimulator
from caussim.experiences.base_config import DEFAULT_SIMU_CONFIG

memory = Memory(DIR2CACHE, verbose=0)

AVAILABLE_DATASETS = ["acic_2018", "caussim", "twins", "acic_2016"]


def load_dataset(dataset_config: Dict) -> Tuple[Union[None, CausalSimulator], CausalDf]:
    dataset_name = dataset_config["dataset_name"]
    assert (
        dataset_name in AVAILABLE_DATASETS
    ), f"Only {AVAILABLE_DATASETS}, got {dataset_name}"
    sim = None
    if dataset_name == "acic_2018":
        causal_df = load_acic_2018_from_ufid(dataset_config["ufid"])
    elif dataset_name == "twins":
        causal_df = load_twins(
            random_state=dataset_config["random_state"],
            overlap=dataset_config["overlap"],
        )
    elif dataset_name == "acic_2016":
        causal_df = load_acic_2016(
            dgp=dataset_config["dgp"], seed=dataset_config["seed"]
        )
    elif dataset_name == "caussim":
        sim = CausalSimulator(
            n_samples_init=dataset_config["n_samples_init"],
            dim=dataset_config["dim"],
            baseline_link=dataset_config["baseline_link"],
            effect_link=dataset_config["effect_link"],
            treatment_link=dataset_config["treatment_link"],
            treatment_assignment=dataset_config["treatment_assignment"],
            effect_size=dataset_config["effect_size"],
            outcome_noise=dataset_config["outcome_noise"],
            treatment_ratio=dataset_config["treatment_ratio"],
            random_seed=dataset_config["random_seed"],
            clip=dataset_config["clip"],
        )
        causal_df = sim.sample(num_samples=dataset_config["n_samples_init"])
    return sim, causal_df


@memory.cache
def load_acic_2018_from_ufid(ufid: str):
    """Load ACIC 2018 data obtained from https://github.com/IBM-HRL-MLHLS/IBM-Causal-Inference-Benchmarking-Framework/tree/master/data/LBIDD

    We only take the scaling part of the data, obtained from [this URL](https://www.synapse.org/#!Synapse:syn11738963)

    References:
        Y. Shimoni, C. Yanover, E. Karavani, et Y. Goldschmnidt, « Benchmarking Framework for Performance-Evaluation of Causal Inference Analysis », arXiv:1802.05046


    Parameters
    ----------
    ufid : str
        _description_

    Returns
    -------
    _type_
        _description_
    """
    X = pd.read_csv(str(PATH2ACIC_2018_X))

    X.columns = [f"X_{col}" if col != "sample_id" else col for col in X.columns]
    X_cols = [col for col in X.columns if col != "sample_id"]

    Y_f = pd.read_csv(str(DIR2ACIC_2018_F / f"{ufid}.csv"))
    Y_cf = pd.read_csv(str(DIR2ACIC_2018_CF / f"{ufid}_cf.csv"))
    df = (
        Y_f.merge(Y_cf, on="sample_id", how="inner")
        .merge(X, on="sample_id", how="inner")
        .rename(columns={"sample_id": "idx", "z": "a", "y0": "y_0", "y1": "y_1"})
        .assign(
            e=0.5 * np.ones(Y_f.shape[0]),
            mu_0=lambda df: df["y_0"],
            mu_1=lambda df: df["y_1"],
        )  # This is fake e to be consistent with other loading functions
    )
    return CausalDf(
        df[CAUSAL_DF_COLUMNS + X_cols], dataset_name="acic_2018", random_state=ufid
    )


def load_acic_2016(dgp, seed, trim: float = None):
    """
    Load one acic 2016 simulation based on their R package, [aciccomp2016](https://github.com/vdorie/aciccomp)
    Pre-requisite:
    ```
    if (require("remotes", quietly = TRUE) == FALSE) {
        install.packages("remotes")
        require("remotes")
    }
    remotes::install_github("vdorie/aciccomp/2016")

    References:
        V. Dorie, J. Hill, U. Shalit, M. Scott, et D. Cervone, « Automated versus do-it-yourself methods for causal inference: Lessons learned from a data analysis competition », arXiv:1707.02641 [stat], juill. 2018, http://arxiv.org/abs/1707.02641

    ```
    Args:
        param (int): simulations parameters in 1:77
        seed (int): random seed in 1:100

    Returns:
        [type]: [description]
    """
    import rpy2.robjects as robjects
    from rpy2.robjects import conversion

    r = robjects.r
    r["source"](
        str(ROOT_DIR / "caussim" / "data" / "preprocessing.R")
    )  # Loading the function we have defined in R.
    load_acic_2016_x_r = robjects.globalenv["load_acic_2016_x"]
    load_acic_2016_y_r = robjects.globalenv["load_acic_2016_y"]

    X = pd.DataFrame(conversion.rpy2py(load_acic_2016_x_r())).transpose()
    X.columns = [f"X_{i}" for i in range(X.shape[1])]
    y = pd.DataFrame(conversion.rpy2py(load_acic_2016_y_r(dgp, seed))).transpose()
    y.columns = ["a", "y", "y_0", "y_1", "mu_0", "mu_1", "e"]

    df = pd.concat((y, X), axis=1)
    if trim is not None:
        df = df.loc[(df["e"] >= trim) & (df["e"] <= 1 - trim), :]
    df.reset_index(drop=True, inplace=True)
    df["idx"] = df.index

    simulation_setups = pd.read_csv(DIR2ACIC_2016 / "simulation_setups.csv")
    overlap = simulation_setups.loc[simulation_setups["dgp"] == dgp, "overlap"].values[
        0
    ]
    if overlap == "penalize":
        overlap = 0
    elif overlap == "full":
        overlap = 1
    else:
        logging.warning(f"Unknown overlap value: {overlap}, putting nan")
        overlap = np.nan
    return CausalDf(
        df,
        dataset_name="acic_2016",
        overlap_parameter=overlap,
        random_state=seed,
        dgp=dgp,
    )


def load_twins(random_state: int = None, overlap: float = 1) -> CausalDf:
    """
    Load the twins dataset obtained from shalit lab.
    Louizos et al. (2017) introduced the Twins dataset as an augmentation of the
    real data on twin births and twin mortality rates in the USA from 1989-1991
    (Almond et al., 2005). The treatment is "born the heavier twin" so, in one
    sense, we can observe both potential outcomes. Louizos et al. (2017) create an
    observational dataset out of this by hiding one of the twins (for each pair) in
    the dataset. To ensure there is some confounding, Louizos et al. (2017)
    simulate the treatment assignment (which twin is heavier) as a function of the
    GESTAT10 covariate, which is the number of gestation weeks prior to birth.
    GESTAT10 is highly correlated with the outcome and it seems intuitive that it
    would be a cause of the outcome, so this should simulate some confounding.
    They simulate this "treatment" with a sigmoid model based on GESTAT10 (number of gestation weeks before birth) and x, the 45 other covariates:
    $\mathbf{t}_{i} \mid \mathbf{x}_{i}, \mathbf{z}_{i} \sim \operatorname{Bern}\left(\sigma\left(w_{o}^{\top} \mathbf{x}+w_{h}(\mathbf{z} / 10-0.1)\right)\right) \quad with \; w_{o} \sim \mathcal{N}(0,0.1 \cdot I), w_{h} \sim \mathcal{N}(5,0.1)$
    Furthermore, to make sure the twins are very similar, they limit
    the data to the twins that are the same sex. To look at data with higher
    mortality rates, they further limit the dataset to twins that were born weighing
    less than 2 kg.

    References:

        Almond, D., Chay, K. Y., & Lee, D. S. (2005). The costs of low birth weight.
            The Quarterly Journal of Economics, 120(3), 1031-1083.

        Louizos, C., Shalit, U., Mooij, J. M., Sontag, D., Zemel, R., & Welling, M.
            (2017). Causal effect inference with deep latent-variable models. In
            Advances in Neural Information Processing Systems (pp. 6446-6456).
        B. Neal, C.-W. Huang, et S. Raghupathi. RealCause: Realistic Causal Inference Benchmarking. arXiv:2011.15007 [cs, stat], march 2021

        Returns:
            CausalDf: _description_
    """
    TWINS_URL = "https://raw.githubusercontent.com/shalit-lab/Benchmarks/master/Twins/Final_data_twins.csv"
    TWINS_FILENAME = "twins.csv"
    local_path_to_raw_twins = DIR2TWINS / TWINS_FILENAME
    if not (DIR2TWINS / TWINS_FILENAME).exists():
        twins_df = pd.read_csv(TWINS_URL)
        DIR2TWINS.mkdir(exist_ok=True, parents=True)
        twins_df.to_csv(local_path_to_raw_twins, index=False)
    else:
        twins_df = pd.read_csv(local_path_to_raw_twins)

    twins_df = twins_df.rename(
        columns={
            "Unnamed: 0": "idx",
            "T": "a",
            "y0": "y_0",
            "y1": "y_1",
            "yf": "y",
            "Propensity": "e",
        }
    ).drop(["y_cf"], axis=1)
    not_covariate_cols = ["idx", "a", "y_0", "y_1", "y", "e"]
    x_cols = [col for col in twins_df.columns if col not in not_covariate_cols]
    x_cols_dic = {col: f"x_{col}" for col in x_cols}
    twins_df = twins_df.rename(columns=x_cols_dic)
    twins_df["mu_0"] = twins_df["y_0"]
    twins_df["mu_1"] = twins_df["y_1"]
    if (random_state is not None) | (overlap != 1):
        ps, a = twins_propensity_model(
            twins_covariates=twins_df[list(x_cols_dic.values())],
            random_state=random_state,
            alpha=overlap,
        )
        twins_df["e"] = ps
        twins_df["a"] = a
        twins_df["y"] = (1 - a) * twins_df["y_0"] + a * twins_df["y_1"]
    return CausalDf(
        twins_df[[*CAUSAL_DF_COLUMNS, *list(x_cols_dic.values())]],
        dataset_name="twins",
        overlap_parameter=overlap,
        random_state=random_state,
    )


def twins_propensity_model(
    twins_covariates: pd.DataFrame, random_state=None, alpha=1
) -> Tuple[np.array, np.array]:
    """Reproduce a logistic propensity model for the twins dataset as o in the paper of Louizos et al., 2017.
    Returns the propensity score and the treatment assignment.

    Args:
        twins_covariates (pd.DataFrame): The full covariates of the twins dataset as given by the shalit-lab.
        random_state (_type_, optional): random number generator. Defaults to None.
        alpha (int, optional): Control the overlap, a value higher than 1 will induce less overlap. Defaults to 1.

    References:
         Louizos, C., Shalit, U., Mooij, J. M., Sontag, D., Zemel, R., & Welling, M.
            (2017). Causal effect inference with deep latent-variable models. In
            Advances in Neural Information Processing Systems (pp. 6446-6456).
    """
    generator = check_random_state(random_state)
    gestatbin_cols = [f"x_gestatcat{b}" for b in np.arange(1, 11)]
    gestat_cols = [
        col for col in twins_covariates.columns if col.find("x_gestatcat") != -1
    ]
    gestat10 = twins_covariates[gestatbin_cols].sum(axis=1) - 1
    ihdp_covs_wo_gestat = twins_covariates.drop(gestat_cols, axis=1)
    w_o = generator.normal(0, 0.1, size=ihdp_covs_wo_gestat.shape[1])
    w_h = generator.normal(5, 0.1, size=1)
    ps = sigmoid(alpha * (ihdp_covs_wo_gestat.dot(w_o) + w_h * (gestat10 / 10 - 0.1)))
    a = generator.binomial(1, ps)
    return ps, a


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# script to create dataset config easily for each dataset


def make_dataset_config(
    dataset_name,
    overlap: float = 1,
    random_state: int = 0,
    dgp: int = 1,
    ufid: str = "a957b431a74a43a0bb7cc52e1c84c8ad",
    treatment_ratio: float = 0.1,
    **kwargs,
) -> Dict:
    """
    Simplify the creation of a dataset config dictionary.

    Parameters
    ----------
    dataset_name : _type_
        _description_
    overlap : float, optional
        Used for caussim and twins, by default 1
    random_state : int, optional
        Used for caussim, acic_2016, twins, by default 0
    dgp : int, optional
        Dgp identifier, used for acic_2016, by default 1
    ufid : str, optional
        Unique identifier of the dataset, used only for acic_2018,  by default "a957b431a74a43a0bb7cc52e1c84c8ad"

    Returns
    -------
    Dict
        _description_
    """
    if dataset_name == "caussim":

        dataset_config = deepcopy(DEFAULT_SIMU_CONFIG)
        dataset_config[dataset_name] = dataset_name
        dataset_config["treatment_assignment"]["params"]["overlap"] = overlap
        dataset_config["treatment_ratio"] = treatment_ratio
        dataset_config["random_seed"] = random_state

        dataset_config.update(**kwargs)
        dataset_config["train_seed"] = dataset_config["random_seed"]
        dataset_config["test_seed"] = dataset_config["random_seed"] + 1
        if "nuisance_set_size" in dataset_config.keys():
            dataset_config["nuisance_set_seed"] = dataset_config["random_seed"] + 2

    elif dataset_name == "twins":
        dataset_config = {
            "dataset_name": dataset_name,
            "random_state": random_state,
            "overlap": overlap,
        }
    elif dataset_name == "acic_2016":
        dataset_config = {
            "dataset_name": dataset_name,
            "dgp": dgp,
            "seed": random_state,
        }
    elif dataset_name == "acic_2018":
        dataset_config = {"dataset_name": dataset_name, "ufid": ufid}
    else:
        raise ValueError(
            f"Does not support {dataset_name} as dataset, please choose one of: {AVAILABLE_DATASETS}"
        )

    return dataset_config

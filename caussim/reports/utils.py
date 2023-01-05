import logging
import pickle
import re
from typing import Dict, List, Union
from pathlib import Path
from typing import Tuple
from joblib import Parallel
from sklearn.utils.fixes import delayed

from matplotlib import pyplot as plt

import numpy as np
from caussim.config import (
    DIR2EXPES,
    DIR2FIGURES,
    DIR2NOTES,
    LOGGER_OUT,
    DIR2PAPER_IMG,
)
import pandas as pd
import yaml
from caussim.pdistances.point_clouds import chamfer_distance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from caussim.config import DIR2FIGURES, TAB_COLORS, DIR2CACHE
from scipy.stats import kendalltau
from tqdm import tqdm


from joblib.memory import Memory

memory = Memory(DIR2CACHE, verbose=0)

"""
Report utils
"""


def read_logs(xp_name: Path) -> Tuple[pd.DataFrame, dict]:
    """Read results from a causal score evaluation experience.

    Parameters
    ----------
    xp_name : Path
        _description_

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        _description_
    """
    xp_type = xp_name.parent.stem
    dir2save = DIR2EXPES / xp_type / xp_name

    run_logs = pd.read_csv(dir2save / "run_logs.csv")
    if (dir2save / "simu.yaml").exists():
        with open(dir2save / "simu.yaml", "r") as f:
            simu_config = yaml.load(f, yaml.FullLoader)
    elif (dir2save / "simu.pkl").exists():
        with open(dir2save / "simu.pkl", "rb") as f:
            simu_config = pickle.load(f)
    else:
        logging.warning("No configuration file for this experience")
        simu_config = None
    # Derive important quantities from the run logs
    run_logs["normalized_abs_bias_ate"] = run_logs["abs_bias_ate"] / np.absolute(
        run_logs["ate"]
    )
    run_logs["test_eta"] = np.min(
        (run_logs["test_e_min"], 1 - run_logs["test_e_max"]), axis=0
    )
    run_logs["check_e_eta"] = np.min(
        (run_logs["check_e_min"], 1 - run_logs["check_e_max"]), axis=0
    )
    if "components" in run_logs.columns:
        compute_distance_to_gold_rpz(run_logs)
    if ("r_risk_ipw" in run_logs.columns) and ("r_risk" in run_logs.columns):
        run_logs["r_risk_IS2"] = run_logs["r_risk_ipw"] + 2 * run_logs["r_risk"]
        run_logs["oracle_r_risk_IS2"] = (
            run_logs["oracle_r_risk_ipw"] + 2 * run_logs["oracle_r_risk"]
        )
    else:
        logging.warning("No r-risk in run logs : Cannot create r_risk_IS2")

    if "treatment_heterogeneity" in run_logs.columns:
        run_logs["treatment_heterogeneity_norm"] = run_logs[
            "treatment_heterogeneity"
        ] / np.abs(run_logs["ate"])
        run_logs["log_treatment_heterogeneity_norm"] = np.log(
            run_logs["treatment_heterogeneity_norm"]
        )
    if ("heterogeneity_score" in run_logs.columns) or (
        "hetereogeneity_score" in run_logs.columns
    ):
        run_logs = run_logs.rename(
            columns={"hetereogeneity_score": "heterogeneity_score"}
        )
        run_logs["heterogeneity_score_norm"] = np.sqrt(
            run_logs["heterogeneity_score"]
        ) / np.abs(run_logs["ate"])
    return run_logs, simu_config


def get_global_agreement_w_tau_risk(comparison_df_w_best_as_oracle: pd.DataFrame):
    n_0_tau_risk_df_by_metric = []
    for metric in comparison_df_w_best_as_oracle["causal_metric"].unique():
        metric_data = comparison_df_w_best_as_oracle.loc[
            comparison_df_w_best_as_oracle["causal_metric"] == metric
        ]
        n_xp_w_0_tau_risk = (
            metric_data["tau_risk"] == metric_data["tau_risk_as_oracle"]
        ).sum()
        n_0_tau_risk_df_by_metric.append(
            {
                "Selection metric": metric,
                "Global agreement with tau_risk (%)": np.round(
                    100 * n_xp_w_0_tau_risk / len(metric_data), 2
                ),
            }
        )
    n_0_tau_risk_df_by_metric = pd.DataFrame(n_0_tau_risk_df_by_metric).sort_values(
        "Selection metric"
    )
    return n_0_tau_risk_df_by_metric


def compute_distance_to_gold_rpz(run_logs):
    """Wrapper around saved logs to compute distance between representers of each model and representers from the simulation.

    Args:
        run_logs (_type_): _description_

    Returns:
        _type_: _description_
    """
    run_logs["simu_components"] = run_logs["simu_components"].apply(
        lambda x: np.array(eval(re.sub("\[,", "[", re.sub("[\s]{1,}|\n", ",", x))))
    )
    run_logs["components"] = run_logs["components"].apply(
        lambda x: np.array(eval(re.sub("\[,", "[", re.sub("[\s]{1,}|\n", ",", x))))
    )
    LOGGER_OUT.info("Computing chamfer distance")
    run_logs["chamfer_distance"] = run_logs.apply(
        lambda df: chamfer_distance(df["simu_components"], df["components"]), axis=1
    )

    return run_logs


def read_multiple_logs(xp_names: List[Path]):
    run_logs = []
    for xp_name in xp_names:
        run_logs.append(read_logs(xp_name)[0])
    return pd.concat(run_logs, axis=0)


@memory.cache
def get_metric_rankings_by_dataset(
    expe_results: pd.DataFrame,
    expe_indices: List[str],
    causal_metrics: List[str],
    candidate_params: List[str] = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Given for different experiments, the evaluation of all candidate estimators by different causal metrics, compute the ranking of each candidate among other candidates of the same experiment.

    Parameters
    ----------
    expe_results : pd.DataFrame
        _description_
    expe_indices : List[str]
        _description_
    causal_metrics : List[str]
        _description_
    candidate_params : List[str], optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """

    if candidate_params == None:
        candidate_params = ["run_id"]

    unique_indices = expe_results[expe_indices].drop_duplicates()

    logging.info(f"{len(unique_indices)} unique experience (dataset x dgp x seed")
    expe_indices_dic = [
        dict(zip(expe_indices, unique_expe_ix))
        for unique_expe_ix in unique_indices.values
    ]
    expe_results_list = []
    for expe_index_dic in expe_indices_dic:
        masking = pd.concat(
            [expe_results[xp_ix] == ix for (xp_ix, ix) in expe_index_dic.items()],
            axis=1,
        ).apply(sum, axis=1) == len(expe_indices)
        single_expe_results = expe_results.loc[masking]
        expe_results_list.append(single_expe_results)

    def _get_metric_rankings_single_xp(
        single_expe_results: pd.DataFrame,
        expe_indices: List[str],
        candidate_params: List[str],
        causal_metrics: List[str],
    ):
        ranking_cols = {
            metric_name: f"ranking_{metric_name}" for metric_name in causal_metrics
        }
        for metric_name in causal_metrics:
            metric_argsort = single_expe_results[metric_name].argsort()
            metric_rank = metric_argsort.argsort()
            single_expe_results = single_expe_results.assign(
                **{ranking_cols[metric_name]: metric_rank}
            )
        return single_expe_results[
            [*expe_indices, *candidate_params, *ranking_cols.values()]
        ]

    parallel = Parallel(n_jobs=n_jobs)
    with parallel:
        out = parallel(
            delayed(_get_metric_rankings_single_xp)(
                single_expe_results=single_expe_results,
                expe_indices=expe_indices,
                candidate_params=candidate_params,
                causal_metrics=causal_metrics,
            )
            for single_expe_results in expe_results_list
        )

    return pd.concat(out, axis=0)


def kendalltau_stats(x, y):
    return kendalltau(x, y)[0]


@memory.cache
def get_rankings_aggregate(
    expe_rankings: pd.DataFrame,
    expe_indices: List[str],
    causal_metrics: List[str],
    n_jobs=-1,
) -> pd.DataFrame:
    """
    Aggregate rankings for each experiment (dataset x dgp x seed)

    Parameters
    ----------
    expe_rankings : pd.DataFrame
        _description_
    expe_indices : List[str]
        _description_
    causal_metrics : List[str]
        _description_
    """

    aggregation_f_ = kendalltau_stats
    unique_indices = expe_rankings[expe_indices].drop_duplicates()
    logging.info(f"{len(unique_indices)} unique experience (dataset x dgp x seed")
    causal_metrics_ = [m for m in causal_metrics if m != "tau_risk"]

    expe_indices_dic = [
        dict(zip(expe_indices, unique_expe_ix))
        for unique_expe_ix in unique_indices.values
    ]
    expe_rankings_list = []
    for expe_index_dic in expe_indices_dic:
        masking = pd.concat(
            [expe_rankings[xp_ix] == ix for (xp_ix, ix) in expe_index_dic.items()],
            axis=1,
        ).apply(sum, axis=1) == len(expe_indices)
        single_expe_rankings = expe_rankings.loc[masking]
        expe_rankings_list.append(single_expe_rankings)

    def _get_rankings_aggregate_single_expe(
        expe_index_dic: Dict, single_expe_rankings: pd.DataFrame
    ):
        # convoluted way to combine the multiple filters
        for metric_name in causal_metrics_:
            expe_index_dic[
                f"{aggregation_f_.__name__}__tau_risk_{metric_name}"
            ] = aggregation_f_(
                single_expe_rankings["ranking_tau_risk"],
                single_expe_rankings[f"ranking_{metric_name}"],
            )
        return expe_index_dic

    parallel = Parallel(n_jobs=n_jobs)
    with parallel:
        out = parallel(
            delayed(_get_rankings_aggregate_single_expe)(
                expe_index_dic=expe_index_dic,
                single_expe_rankings=single_expe_rankings,
            )
            for (expe_index_dic, single_expe_rankings) in zip(
                expe_indices_dic, expe_rankings_list
            )
        )
    return pd.DataFrame(out)


def get_best_estimator_by_dataset(
    expe_logs,
    expe_indices: List[str],
    candidate_params: List[str],
    causal_metrics: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Return a dataframe with the best method for each simulation.
    A simulation is uniquely defined by the indices and comport several lines in the input dataframe : one line per candidate model.
    The information on the best candidate model kept in the resulting dataframe is the model_params List.
    We do that for each causal metrics and return a dict.

    Parameters
    ----------
    run_logs : _type_
        _description_
    indices : List[str]
        _description_
    model_params : List[str]
        _description_
    causal_metrics : List[str]
        _description_

    Returns
    -------
    Dict[str, pd.DataFrame]
        _description_
    """
    logging.info("Nb xps : %f", len(expe_logs[expe_indices].drop_duplicates()))
    evaluation_metrics = [
        eval_m
        for eval_m in [
            "mse_ate",
            "abs_bias_ate",
            "normalized_abs_bias_ate",
            "check_m_r2",
            "check_e_bss",
            "check_e_eta",
            "check_e_auroc",
            "check_e_mse",
            "check_e_IPW_mse",
            "check_e_inv_mse",
            "ate",
        ]
        if eval_m in expe_logs.columns
    ]

    best_methods_by_xp = {}
    for metric in causal_metrics:
        best_methods_by_xp[metric] = (
            expe_logs.sort_values(metric, ascending=True)
            .groupby(expe_indices)
            .agg("first")[
                ["run_id", *causal_metrics, *evaluation_metrics, *candidate_params]
            ]
            .reset_index()
            .assign(causal_metric=metric)
        )
    best_methods_by_xp["random"] = (
        expe_logs.sample(frac=1, random_state=0)
        .groupby(expe_indices)
        .agg("first")[
            ["run_id", *causal_metrics, *evaluation_metrics, *candidate_params]
        ]
        .reset_index()
    )
    best_methods_by_xp["random"]["causal_metric"] = "random"
    return best_methods_by_xp


def get_nuisances_type(df):
    """Extract labels of the type of estimator used for nuisance estimation in causal score experiments.

    Parameters
    ----------
    df : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    non_linearmodels_regex = "|".join(["gradientboosting", "randomforest"])
    nuisance_estimator_col = "nuisance_y_estimator"
    if nuisance_estimator_col not in df.columns:
        nuisance_estimator_col = "y_estimator"
        if "y_estimator" not in df.columns:
            return "linear"

    if (
        re.search(non_linearmodels_regex, df[nuisance_estimator_col][0].strip().lower())
        is not None
    ):
        model_label = "non linear"
    else:
        model_label = "linear"
    return model_label


def save_figure_to_folders(
    figure_name: Union[str, Path],
    figure_dir=False,
    paper_dir=False,
    notes_dir=False,
    pdf=True,
):
    figure_name = Path(figure_name)
    reference_folder = figure_name.parents[0].stem
    if figure_dir:
        (DIR2FIGURES / reference_folder).mkdir(exist_ok=True, parents=True)
        plt.savefig(
            str(DIR2FIGURES / reference_folder / f"{figure_name.name}.png"),
            bbox_inches="tight",
        )
        if pdf:
            plt.savefig(
                str(DIR2FIGURES / reference_folder / f"{figure_name.name}.pdf"),
                bbox_inches="tight",
            )
    if paper_dir:
        (DIR2PAPER_IMG / reference_folder).mkdir(exist_ok=True, parents=True)
        plt.savefig(
            str(DIR2PAPER_IMG / reference_folder / f"{figure_name.name}.png"),
            bbox_inches="tight",
        )
        if pdf:
            plt.savefig(
                str(DIR2PAPER_IMG / reference_folder / f"{figure_name.name}.pdf"),
                bbox_inches="tight",
            )
    if notes_dir:
        (DIR2NOTES / reference_folder).mkdir(exist_ok=True, parents=True)
        plt.savefig(
            str(DIR2NOTES / reference_folder / f"{figure_name.name}.png"),
            bbox_inches="tight",
        )
        if pdf:
            plt.savefig(
                str(DIR2NOTES / reference_folder / f"{figure_name.name}.pdf"),
                bbox_inches="tight",
            )

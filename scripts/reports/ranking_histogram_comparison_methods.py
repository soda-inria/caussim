#
import re
from typing import Tuple
import pandas as pd
import numpy as np

import pytest
from caussim.reports import (
    read_logs,
    save_figure_to_folders,
    get_nuisances_type,
)
from caussim.reports.plots_utils import (
    CAUSAL_METRICS, plot_metric_rankings_by_overlap_bin)
from caussim.reports.utils import get_metric_rankings_by_dataset, get_rankings_aggregate, get_expe_indices, get_candidate_params, kendalltau_stats
from caussim.utils import *
from caussim.config import *

sns.set_style("whitegrid")

@pytest.mark.parametrize(
    "xp_paths, reference_metric, comparison_label, plot_xlabel, plot_legend",
    [
        # (
        #     {
        #         "Two sets": Path(DIR2EXPES / "caussim_save"/ "caussim__nuisance_non_linear__candidates_ridge__overlap_01-247_join_nuisance_train_set"),
        #         "Three sets":Path(
        #         DIR2EXPES
        #         / "caussim_save"
        #         / "caussim__nuisance_non_linear__candidates_ridge__overlap_01-247_separated_nuisance_train_set"),
        #     }, "mean_risks", "Training procedure", True, True
        #     ),
        (
            {
                "Linear": Path(DIR2EXPES / "caussim_save"/ "caussim__linear_regressor__test_size_5000__n_datasets_1000"),
                "Stacked": Path(
                DIR2EXPES
                / "caussim_save"
                / "caussim__stacked_regressor__test_size_5000__n_datasets_1000"),
            }, "mean_risks", "Nuisance models", True, True
            )
    ],
)
def test_report_causal_scores_evaluation(
    xp_paths: Path,
    reference_metric: str,
    comparison_label: str,
    plot_xlabel: bool,
    plot_legend: bool
):
    expe_results = {}
    for expe_name, xp_path in xp_paths.items():
        xp_res_, _ = read_logs(xp_path)
        # # subsetting
        # subset_of_xp = xp_res_["test_d_normalized_tv"].value_counts().index[:100]
        # xp_res_ = xp_res_.loc[
        #     xp_res_["test_d_normalized_tv"].isin(subset_of_xp)
        # ]
        expe_results[expe_name] = xp_res_
    # for all datasets
    expe_causal_metrics = [
        metric for metric in CAUSAL_METRICS if metric in xp_res_.columns if metric !="mu_risk"
    ]
    dataset_name = xp_res_["dataset_name"].values[0]
    if dataset_name == "acic_2018":
        xp_res_["test_d_normalized_tv"] = xp_res_["hat_d_normalized_tv"]
        expe_causal_metrics = [
            m for m in expe_causal_metrics if re.search("oracle|gold", m) is None
        ]
    #nuisance_models_label = get_nuisances_type(xp_res_)
    candidate_params = get_candidate_params(xp_res_)
    overlap_measure, expe_indices = get_expe_indices(xp_res_)
        
    logging.info(
        f"Nb (simulations x test_seed) : {len(xp_res_[expe_indices].drop_duplicates())}"
    )
    g = plot_metric_rankings_by_overlap_bin(
        expe_results=expe_results,
        reference_metric=reference_metric,
        expe_indices=expe_indices,
        expe_causal_metrics=expe_causal_metrics,
        candidate_params=candidate_params,
        overlap_measure=overlap_measure,
        comparison_label=comparison_label
    )
    if not plot_xlabel:
        g.fig.suptitle("")
    if reference_metric is not None:
        #ylim_ranking = (ylim_ranking[0] - ylim_ranking[1], ylim_ranking[1])
        ref_metric_str = f"_ref_metric_{reference_metric}"
    else:
        ref_metric_str = ""
    plot_name = "_".join(comparison_label.lower().split(" "))
    if not plot_legend:
        g._legend.remove()
    save_figure_to_folders(
        figure_name=Path(f"kendalls_tau{ref_metric_str}_bin/{dataset_name}_{plot_name}"),
        figure_dir=True,
        notes_dir=False,
        paper_dir=True,
    )
    return g

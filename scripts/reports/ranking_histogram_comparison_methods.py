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
    CAUSAL_METRICS)
from caussim.reports.utils import get_metric_rankings_by_dataset, get_rankings_aggregate, get_expe_indices, get_candidate_params, kendalltau_stats
from caussim.utils import *
from caussim.config import *

sns.set_style("whitegrid")

@pytest.mark.parametrize(
    "xp_paths, reference_metric, ylim_ranking, plot_name, comparison_label",
    [
        (
            {
                "Two sets": Path(DIR2EXPES / "caussim_save"/ "caussim__nuisance_non_linear__candidates_ridge__overlap_01-247_join_nuisance_train_set"),
                "Three sets":Path(
                DIR2EXPES
                / "caussim_save"
                / "caussim__nuisance_non_linear__candidates_ridge__overlap_01-247_separated_nuisance_train_set"),
            }, None, (0, 1.05), "binned_comparison_kendall_tau_joint_vs_separated_nuisance_train_set", "Training procedure"
            )
    ],
)
def test_report_causal_scores_evaluation(
    xp_paths: Path,
    reference_metric: str,
    ylim_ranking: Tuple[float, float],
    plot_name: str,
    comparison_label: str
):
    expe_results = {}
    for expe_name, xp_path in xp_paths.items():
        xp_res_, _ = read_logs(xp_path)
        # subsetting
        subset_of_xp = xp_res_["test_d_normalized_tv"].value_counts().index[:100]
        xp_res_ = xp_res_.loc[
            xp_res_["test_d_normalized_tv"].isin(subset_of_xp)
        ]
        expe_results[expe_name] = xp_res_
    # for all datasets
    xp_causal_metrics = [
        metric for metric in CAUSAL_METRICS if metric in xp_res_.columns
    ]
    dataset_name = xp_res_["dataset_name"].values[0]

    if dataset_name == "acic_2018":
        xp_res_["test_d_normalized_tv"] = xp_res_["hat_d_normalized_tv"]
    nuisance_models_label = get_nuisances_type(xp_res_)
    candidate_params = get_candidate_params(xp_res_)
    overlap_measure, expe_indices = get_expe_indices(xp_res_)
        
    logging.info(
        f"Nb (simulations x test_seed) : {len(xp_res_[expe_indices].drop_duplicates())}"
    )
    if reference_metric is not None:
        ylim_ranking = (ylim_ranking[0] - ylim_ranking[1], ylim_ranking[1])
        ref_metric_str = f"_ref_metric_{reference_metric}"
    else:
        ref_metric_str = ""
    binned_results = []
    for xp_name, xp_res in expe_results.items():
        expe_rankings = get_metric_rankings_by_dataset(
            expe_results=xp_res,
            expe_indices=expe_indices,
            causal_metrics=xp_causal_metrics,
            candidate_params=candidate_params,
        )

        rankings_agg = get_rankings_aggregate(
            expe_rankings=expe_rankings,
            expe_indices=expe_indices,
            causal_metrics=xp_causal_metrics,
        )
    
        # Group by experience and plot results as binned boxplots 
        aggregation_f_name = kendalltau_stats.__name__
        evaluation_metric = aggregation_f_name
        rankings_matches = [
            re.search(f"{aggregation_f_name}__tau_risk_(.*)", col)
            for col in rankings_agg.columns
        ]
        selection_metrics = [
            reg_match.group(1) for reg_match in rankings_matches if reg_match is not None
        ]
        rankings_name = [
            reg_match.group(0) for reg_match in rankings_matches if reg_match is not None
        ]
        var_name = "causal_metric"
        rankings_aggregation_melted = rankings_agg.melt(
            id_vars=expe_indices,
            value_vars=rankings_name,
            var_name=var_name,
            value_name=evaluation_metric,
        )
        # shape = n_experiences x n_causal_metrics
        bins_quantiles = [0,0.33, 0.66, 1]
        bins_values = rankings_aggregation_melted[overlap_measure].quantile(bins_quantiles).values
        bins_labels = [f"{b_low:.2f}-{b_sup:.2f}" for b_low, b_sup in zip(bins_values[:-1], bins_values[1:])]
        rankings_aggregation_melted["overlap_bin"] = pd.cut(rankings_aggregation_melted[overlap_measure], bins=bins_values, labels=bins_labels)
        rankings_aggregation_melted[comparison_label] = xp_name
        binned_results.append(rankings_aggregation_melted)
    binned_results_df = pd.concat(binned_results, axis=0)
    breakpoint()
    # TODO: seaborn boxplots
    plt.tight_layout()

    save_figure_to_folders(
        figure_name=Path(f"kendalls_tau{ref_metric_str}_bin/{dataset_name}_{plot_name}"),
        figure_dir=True,
        notes_dir=False,
        paper_dir=True,
    )

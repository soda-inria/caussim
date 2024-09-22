#
import re
from typing import Tuple
from matplotlib.patches import Patch
from matplotlib.text import Text
import pandas as pd
import numpy as np

import pytest
from caussim.reports import (
    read_logs,
    save_figure_to_folders,
)
from caussim.reports.plots_utils import (
    CAUSAL_METRICS, DATASETS_PALETTE, EVALUATION_METRIC_LABELS, MAP_DATASET_LABEL, OVERLAP_BIN_COL, OVERLAP_BIN_LABELS, get_kendall_by_overlap_bin, plot_metric_rankings_by_overlap_bin,DATASET_LABEL, PROCEDURE_LABEL, PROCEDURE_PALETTE, SEPARATED_SET_LABEL, SHARED_SET_LABEL)
from caussim.reports.utils import get_expe_indices
from caussim.utils import *
from caussim.config import *

sns.set_style("whitegrid")


PROCEDURE_EXPERIMENTS = [
            (
            {
                SHARED_SET_LABEL: Path(DIR2EXPES / "caussim_save"/ "caussim__nuisance_non_linear__candidates_ridge__overlap_01-247_join_nuisance_train_set.parquet"),
                SEPARATED_SET_LABEL:Path(
                DIR2EXPES
                / "caussim_save"
                / "caussim__nuisance_non_linear__candidates_ridge__overlap_01-247_separated_nuisance_train_set.parquet"),
            }, "oracle_r_risk", PROCEDURE_LABEL, False, True, False
            ),
            (
            {
            SHARED_SET_LABEL: Path(DIR2EXPES / "acic_2016_save"/ "acic_2016__nuisance_non_linear__candidates_hist_gradient_boosting__dgp_1-77__rs_1-10.parquet"),
                SEPARATED_SET_LABEL:Path(
                DIR2EXPES
                / "acic_2016_save"
                / "acic_2016__nuisance_non_linear__candidates_hist_gradient_boosting__3set_050__dgp_1-77__rs_1-10.parquet"),
            }, "oracle_r_risk", PROCEDURE_LABEL, False, False, False
            ),
            (
            {
            SHARED_SET_LABEL: Path(DIR2EXPES / "twins_save"/ "twins__full_nuisance_stacking__candidates_hist_gradient_boosting__tset_50__overlap_11-296__rs_0-9.parquet"),
                SEPARATED_SET_LABEL:Path(
                DIR2EXPES
                / "twins_save"
                / "twins__nuisance_non_linear__candidates_hist_gradient_boosting__3set_050__overlap_11-296__rs_0-9.parquet"),
            }, "oracle_r_risk", PROCEDURE_LABEL, True, False, False
            ),
]

@pytest.mark.parametrize(
    "xp_paths, reference_metric, comparison_label, plot_xlabel, plot_legend, plot_middle_bin",
    [
        *PROCEDURE_EXPERIMENTS
    ],
)
def test_report_causal_scores_evaluation(
    xp_paths: Path,
    reference_metric: str,
    comparison_label: str,
    plot_xlabel: bool,
    plot_legend: bool,
    plot_middle_bin: bool
):
    """
    Plot supporting the argument that there is no need to use a 3 set procedure
    with a separated nuisance set from the canidates training set.

    Args:
        xp_paths (Path): Paths of the experiments to compare
        reference_metric (str): the metric to use as a reference for the ranking
        (default is the mean of kendalls over all metrics)
        comparison_label (str): the hue legend title (ie. main axis of analysis)
        plot_xlabel (bool): 
        plot_legend (bool): 
        plot_middle_bin (bool): if True, the middle tertile of overlap is plotted.

    Returns:
        _type_: _description_
    """
    expe_results = {}
    for expe_name, xp_path in xp_paths.items():
        xp_res_, _ = read_logs(xp_path)
        # subsetting
        # subset_of_xp = xp_res_["test_d_normalized_tv"].value_counts().index[:100]
        # xp_res_ = xp_res_.loc[
        #     xp_res_["test_d_normalized_tv"].isin(subset_of_xp)
        # ]
        expe_results[expe_name] = xp_res_
    # for all datasets
    expe_causal_metrics = [
        metric for metric in CAUSAL_METRICS if ((metric in xp_res_.columns))
    ]
    dataset_name = MAP_DATASET_LABEL[xp_res_["dataset_name"].values[0]]
    g = plot_metric_rankings_by_overlap_bin(
        expe_results=expe_results,
        reference_metric=reference_metric,
        expe_causal_metrics=expe_causal_metrics,
        comparison_label=comparison_label,
        plot_middle_overlap_bin=plot_middle_bin
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
    dataset_name_str = dataset_name.replace("\n", "_")
    save_figure_to_folders(
        figure_name=Path(f"_3_procedure/{dataset_name_str}_{ref_metric_str}_{plot_name}"),
        figure_dir=True,
        notes_dir=False,
        paper_dir=True,
    )
    return g


def test_ranking_r_risk_only():
    all_expe_results_by_bin = []
    
    for dataset_config in PROCEDURE_EXPERIMENTS:
        for procedure_type, xp_path in dataset_config[0].items():
            xp_res_, _ = read_logs(xp_path)
            expe_causal_metrics = [
        metric for metric in CAUSAL_METRICS if ((metric in xp_res_.columns) and (metric not in ["mu_risk", "mu_iptw_risk", "oracle_mu_iptw_risk","oracle_u_risk", "u_risk", "w_risk", "oracle_w_risk"]))
    ]
            kendall_by_overlap_bin, evaluation_metric = get_kendall_by_overlap_bin(
                xp_res=xp_res_,
                expe_causal_metrics=expe_causal_metrics,
                reference_metric="oracle_r_risk",
                plot_middle_overlap_bin=False,
            )
            kendall_by_overlap_bin[PROCEDURE_LABEL] = procedure_type
            kendall_by_overlap_bin[DATASET_LABEL] = MAP_DATASET_LABEL[xp_res_["dataset_name"].values[0]]
            all_expe_results_by_bin.append(kendall_by_overlap_bin)
    all_expe_results_by_bin_df = pd.concat(all_expe_results_by_bin)
    
    ds_used = all_expe_results_by_bin_df[DATASET_LABEL].unique()
    ds_order = [ds_label_ for ds_label_ in DATASETS_PALETTE.keys() if ds_label_ in ds_used]

    g = sns.FacetGrid(
        data=all_expe_results_by_bin_df,
        height=6,
        aspect=1.8,
        col=OVERLAP_BIN_COL, 
        col_order=[OVERLAP_BIN_LABELS[0], OVERLAP_BIN_LABELS[2]])

    g = g.map_dataframe(
        sns.boxplot,
        x=evaluation_metric,
        y=DATASET_LABEL,
        order=ds_order,
        hue=PROCEDURE_LABEL,
        notch=True,
        palette=PROCEDURE_PALETTE,
    )
    g.set_titles(col_template="{col_name}")
    xlabel = EVALUATION_METRIC_LABELS["kendalltau_stats__r_risk__ref_oracle_r_risk_long"]
    g.set(xlabel="",ylabel="", xlim=(-0.75, 0.5))
    g.fig.text(x=0.35, y=0.05, s=xlabel,ha="center", va="center")
    # prune last xtick label
    for a in g.axes.flat:
        # get all the labels of this axis
        labels = a.get_xticklabels()
        # remove the first and the last labels
        labels[0] = labels[-1] = ""
        # set these new labels
        a.set_xticklabels(labels)
    
    legend_handles = [
        plt.Line2D([0], [0], color="white")] + [
        Patch(facecolor=facecolor_, edgecolor="black") for facecolor_ in PROCEDURE_PALETTE.values()
    ]
    legend_labels = [PROCEDURE_LABEL, *list(PROCEDURE_PALETTE.keys())]
    g.add_legend(
        title="", 
        handles=legend_handles, 
        labels=legend_labels, 
        loc="lower center", ncol=3, bbox_to_anchor=(0.35,0.95)
        )
    #breakpoint()
    save_figure_to_folders(
        figure_name=Path(f"_3_procedure/r_risk_only_3datasets"),
        figure_dir=True,
        notes_dir=False,
        paper_dir=True,
    )
    return g
#
import re
from typing import Tuple
import pandas as pd
import numpy as np

import pytest
from caussim.reports import (
    read_logs,
    save_figure_to_folders,
)
from caussim.reports.plots_utils import (
    ACIC_16_LABEL,
    ACIC_18_LABEL,
    CAUSAL_METRIC_LABELS,
    CAUSAL_METRICS,
    CAUSSIM_LABEL,
    DATASETS_PALETTE,
    EVALUATION_METRIC_LABELS,
    FEASIBLE_LABEL,
    METRIC_LABEL,
    METRIC_ORDER,
    METRIC_PALETTE,
    METRIC_PALETTE_BOX_PLOTS,
    METRIC_TYPE,
    ORACLE_PALETTE,
    SEMI_ORACLE_LABEL,
    TWINS_LABEL,
    get_data_from_facetgrid_boxplot,
    get_kendall_by_overlap_bin,
    OVERLAP_BIN_COL,
    OVERLAP_BIN_LABELS,
    DATASET_LABEL,
)
from caussim.utils import *
from caussim.config import *

sns.set_style("whitegrid")

script_name = "_1_r_risk_domination"

DATASET_EXPERIMENTS = [
    (
        {
            CAUSSIM_LABEL: Path(
                DIR2EXPES
                / "caussim_save"
                / "caussim__nuisance_non_linear__candidates_ridge__overlap_01-247.parquet"
            ),
            TWINS_LABEL: Path(
                DIR2EXPES
                / "twins_save"
                / "twins__nuisance_stacking__candidates_hist_gradient_boosting__tset_50__overlap_11-296__rs_0-9_noise10.parquet"
            ),
            ACIC_16_LABEL: Path(
                DIR2EXPES
                / "acic_2016_save"
                / "acic_2016__nuisance_non_linear__candidates_hist_gradient_boosting__dgp_1-77__rs_1-10.parquet"
            ),
            ACIC_18_LABEL: Path(
                DIR2EXPES
                / "acic_2018_save"
                / "acic_2018__nuisance_stacking__candidates_hist_gradient_boosting__tset_50.parquet"
            ),
        },
        "mean_risks",
        True,
        False,
        (-1, 0.7),
    ),
]


@pytest.mark.parametrize(
    "xp_paths, reference_metric, plot_legend, plot_middle_bin, xlim",
    [*DATASET_EXPERIMENTS],
)
def test_report_causal_scores_evaluation(
    xp_paths: Path,
    reference_metric: str,
    plot_legend: bool,
    plot_middle_bin: bool,
    xlim: Tuple[float, float],
):
    """
    For each dataset, and each metric, relative Kendall’s τ agreement with τ -risk, measured as the difference
    between each metric Kendall’s τ and the mean kendall’s τ over all metric:
    κ(`, τ −Risk) − mean`(κ(`, τ −Risk)).

    Args:
        xp_paths (Path): Paths of the experiments to compare
        reference_metric (str): the metric to use as a reference for the ranking
        (default is the mean of kendalls over all metrics)
        plot_legend (bool):
        plot_middle_bin (bool): if True, the middle tertile of overlap is plotted.

    Returns:
        _type_: _description_
    """
    all_expe_results_by_bin = []
    for expe_name, xp_path in xp_paths.items():
        # for all datasets
        xp_res_, _ = read_logs(xp_path)
        expe_causal_metrics = [
            metric for metric in CAUSAL_METRICS if ((metric in xp_res_.columns))
        ]
        dataset_name = xp_res_["dataset_name"].values[0]
        if dataset_name == "acic_2018":
            xp_res_["test_d_normalized_tv"] = xp_res_["hat_d_normalized_tv"]
            expe_causal_metrics = [
                m for m in expe_causal_metrics if re.search("oracle|gold", m) is None
            ]
        # # subsetting
        # subset_of_xp = xp_res_["test_d_normalized_tv"].value_counts().index[:100]
        # xp_res_ = xp_res_.loc[
        #     xp_res_["test_d_normalized_tv"].isin(subset_of_xp)
        # ]
        kendall_by_overlap_bin, evaluation_metric = get_kendall_by_overlap_bin(
            xp_res=xp_res_,
            reference_metric=reference_metric,
            expe_causal_metrics=expe_causal_metrics,
            plot_middle_overlap_bin=plot_middle_bin,
        )
        kendall_by_overlap_bin[DATASET_LABEL] = expe_name
        all_expe_results_by_bin.append(kendall_by_overlap_bin)
    all_expe_results_by_bin_df = pd.concat(all_expe_results_by_bin)

    metric_order = [
        CAUSAL_METRIC_LABELS[metric_label_] for metric_label_ in METRIC_ORDER
    ]
    ds_used = all_expe_results_by_bin_df[DATASET_LABEL].unique()
    hue_order = [
        ds_label_ for ds_label_ in DATASETS_PALETTE.keys() if ds_label_ in ds_used
    ]
    ds_palette = {
        ds_label_: color_
        for (ds_label_, color_) in DATASETS_PALETTE.items()
        if ds_label_ in ds_used
    }

    g = sns.FacetGrid(
        data=all_expe_results_by_bin_df,
        col=OVERLAP_BIN_COL,
        col_order=[OVERLAP_BIN_LABELS[0], OVERLAP_BIN_LABELS[2]],
        height=12,
        aspect=0.6,
    )
    box_alpha = 0.8
    g = g.map_dataframe(
        sns.boxplot,
        x=evaluation_metric,
        y=METRIC_LABEL,
        order=metric_order,
        # hue=METRIC_TYPE,
        palette=ORACLE_PALETTE.values(),
        showfliers=False,
        linewidth=6,
        boxprops={"linewidth": 0, "alpha": box_alpha},
        whiskerprops={"alpha": box_alpha, "color": "grey", "linewidth": 2},
        capprops={"alpha": box_alpha, "color": "grey", "linewidth": 2},
    )
    # add a box by metric for each dataset
    boxplot_data_args = dict(
        x=evaluation_metric,
        y=METRIC_LABEL,
        order=metric_order,
        hue=DATASET_LABEL,
        hue_order=hue_order,
    )
    g = g.map_dataframe(
        sns.boxplot,
        **boxplot_data_args,
        palette=ds_palette,
        notch=True,
        width=0.6,
        capwidths=0.1,
        medianprops={"linewidth": 6},
        # boxprops={"alpha": 1}
    )

    boxplot_df = get_data_from_facetgrid_boxplot(
        data=g.data, **boxplot_data_args, col=g._col_var, col_order=g.col_names
    )
    # breakpoint()
    pretty_boxplot_df = boxplot_df.style.format(precision=2).to_latex(
        hrules=True, clines="all;data"
    )
    with open(DIR2PAPER_IMG / f"{script_name}_boxplot_df.tex", "w") as f:
        f.write(pretty_boxplot_df)
    if plot_legend:
        g.add_legend()
        legend_data = {k: v for k, v in g._legend_data.items() if k in hue_order}
        g._legend.remove()
        g.add_legend(
            title="",
            legend_data=legend_data,
            ncol=4, 
            bbox_to_anchor=(0.5, 1.02), columnspacing=0.5,
            prop={"size": 32},
        )
    else:
        g._legend.remove()
    # renforce the median
    g = g.map_dataframe(
        sns.boxplot,
        x=evaluation_metric,
        y=METRIC_LABEL,
        hue=DATASET_LABEL,
        hue_order=hue_order,
        order=metric_order,
        width=0.8,
        capwidths=0.1,
        showbox=False,
        showcaps=False,
        showfliers=False,
        notch=True,
        whiskerprops={"linewidth": 0},
        medianprops={"linewidth": 6, "color": "black"},
    )
    g.set_titles(col_template="{col_name}", size=36)
    g.set(xlabel="", ylabel="", xlim=xlim)
    if f"{evaluation_metric}_long" in EVALUATION_METRIC_LABELS.keys():
        suptitle = EVALUATION_METRIC_LABELS[f"{evaluation_metric}_long"]
    else:
        suptitle = EVALUATION_METRIC_LABELS[evaluation_metric]
    g.fig.text(s=suptitle, x=0.1, y=0.03)
    if reference_metric is not None:
        ref_metric_str = f"_ref_metric_{reference_metric}"
    else:
        ref_metric_str = "kendall"

    # for lh in g._legend.legendHandles:
    #     lh.set_alpha(box_alpha)
    save_figure_to_folders(
        figure_name=Path(
            f"{script_name}/r_risk_domination_{ref_metric_str}_by_{DATASET_LABEL}"
        ),
        figure_dir=True,
        notes_dir=False,
        paper_dir=True,
    )
    return g

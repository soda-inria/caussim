import re
from typing import List, Tuple
import pandas as pd
import numpy as np

import pytest
from caussim.reports import (
    plot_agreement_w_tau_risk,
    get_best_estimator_by_dataset,
    plot_evaluation_metric,
    read_logs,
    save_figure_to_folders,
    get_nuisances_type,
    get_candidate_params,
    get_expe_indices
)
from caussim.reports.plots_utils import (
    CAUSAL_METRIC_LABELS,
    CAUSAL_METRICS,
    CAUSSIM_LABEL,
    DATASETS_PALETTE,
    EVALUATION_METRIC_LABELS,
    METRIC_LABEL,
    METRIC_ORDER,
    EFFECT_RATIO_BIN_COL,
    EFFECT_RATIO_BIN_PALETTE,
    get_kendall_by_effect_ratio_bin,
    plot_kendall_compare_vs_measure,
    plot_metric_legend,
    DATASET_LABEL, 
    EFFECT_RATIO_BIN_LABELS
)
from caussim.utils import *
from caussim.config import *

sns.set_style("whitegrid")

script_name = "_6_effect_ratio_influence"

EFFECT_RATIO_INFLUENCE_EXPES = [
    (
        Path(
            DIR2EXPES
            / "caussim_save"
            / "caussim__nuisance_linear__candidates_ridge__overlap_06-224.parquet"
        ),
        False,
        0.6,
        (1e-3, 5 * 1e0),
        (0, 1.05),
        None,
    ),
]

@pytest.mark.parametrize(
    "xp_path, show_legend, quantile, ylim_bias_to_tau_risk, ylim_ranking, reference_metric",
    [
        *EFFECT_RATIO_INFLUENCE_EXPES,
    ]
)
def test_report_causal_scores_evaluation(
    xp_path: Path,
    show_legend: bool,
    quantile: float,
    ylim_bias_to_tau_risk: Tuple[float, float],
    ylim_ranking: Tuple[float, float],
    reference_metric: str,
):
    # parameter to select what to be plotted:
    RANKING_COMPUTE = True
    BEST_ESTIMATOR_PLOT = True
    expe_results, _ = read_logs(xp_path)
    dataset_name = expe_results["dataset_name"].values[0]

    # TODO: Maybe remove some metrics here
    expe_causal_metrics = [
        metric for metric in CAUSAL_METRICS if metric in expe_results.columns
    ]
    if dataset_name == "acic_2018":
        expe_causal_metrics = [
            m for m in expe_causal_metrics if re.search("oracle|gold", m) is None
        ]
        expe_results["test_d_normalized_tv"] = expe_results["hat_d_normalized_tv"]
    
    nuisance_models_label = get_nuisances_type(expe_results)
    candidate_params = get_candidate_params(expe_results)
    _, expe_indices = get_expe_indices(expe_results)

    effect_ratio_measure = "effect_ratio"
    expe_indices+=[effect_ratio_measure]
    # ### Exclude some runs with extreme values ### #
    max_mse_ate = 1e5
    outliers_mask = expe_results["mse_ate"] >= max_mse_ate
    outliers = expe_results.loc[outliers_mask]
    expe_results = expe_results.loc[~outliers_mask]
    logging.warning(f"Removing {outliers_mask.sum()} with mse_ate>{max_mse_ate}")
    logging.info(
        f"Nb (simulations x test_seed) : {len(expe_results[expe_indices].drop_duplicates())}"
    )
    if BEST_ESTIMATOR_PLOT:
        best_estimator_by_dataset = get_best_estimator_by_dataset(
            expe_logs=expe_results,
            expe_indices=expe_indices,
            candidate_params=candidate_params,
            causal_metrics=expe_causal_metrics,
        )

        comparison_df_w_random = pd.concat(
            [
                best_estimator_by_dataset[metric]
                for metric in [*expe_causal_metrics, "random"]
            ]
        )
        comparison_df_w_best_as_oracle = comparison_df_w_random.merge(
            best_estimator_by_dataset["tau_risk"][
                ["tau_risk", "run_id", *expe_indices]
            ],
            on=expe_indices,
            how="left",
            suffixes=("", "_as_oracle"),
        )
        comparison_df_w_best_as_oracle["normalized_bias_tau_risk_to_best_method"] = (
            np.abs(
                comparison_df_w_best_as_oracle["tau_risk"]
                - comparison_df_w_best_as_oracle["tau_risk_as_oracle"]
            )
            / comparison_df_w_best_as_oracle["tau_risk_as_oracle"]
        )
        ax, n_found_oracles_grouped_overlap = plot_agreement_w_tau_risk(
            comparison_df_w_best_as_oracle,
            effect_ratio_measure,
            nuisance_models_label=nuisance_models_label,
            show_legend=show_legend,
        )
        plt.tight_layout()
        save_figure_to_folders(
            figure_name=Path(f"agreement_w_tau_risk/{xp_path.stem}"),
            figure_dir=True,
            notes_dir=False,
            paper_dir=True,
        )

        # ### Main figure ### #
        evaluation_measures = ["normalized_bias_tau_risk_to_best_method"]
        for eval_m in evaluation_measures:
            ax = plot_evaluation_metric(
                comparison_df_w_best_as_oracle=comparison_df_w_best_as_oracle,
                evaluation_metric=eval_m,
                measure_of_interest_name=effect_ratio_measure,
                selection_metrics=expe_causal_metrics,
                lowess_type="lowess_quantile",
                lowess_kwargs={"frac": 0.6, "num_fits": 10, "quantile": quantile},
                nuisance_models_label=None,
                ylog=True,
                show_legend=show_legend,
                plot_lowess_ci=False,
            )

            ax.set(ylim=ylim_bias_to_tau_risk)
            plt.tight_layout()
            save_figure_to_folders(
                figure_name=Path(f"{eval_m}/{xp_path.stem}"),
                figure_dir=True,
                notes_dir=False,
                paper_dir=True,
            )

    if RANKING_COMPUTE:
        if reference_metric is not None:
            ref_metric_str = f"_ref_metric_{reference_metric}"
        else:
            ref_metric_str = ""               
        plot_kendall_compare_vs_measure(
            expe_results=expe_results,
            measure="effect_ratio",
            reference_metric=reference_metric,
            expe_causal_metrics=expe_causal_metrics,
            quantile=quantile,
            ylim_ranking=ylim_ranking
        )
        plt.tight_layout()

        save_figure_to_folders(
            figure_name=Path(f"kendalls_tau{ref_metric_str}/{xp_path.stem}"),
            figure_dir=True,
            notes_dir=False,
            paper_dir=True,
        )


DATASET_EXPERIMENTS = [
            (
            {
            CAUSSIM_LABEL: Path(DIR2EXPES / "caussim_save"/ "caussim__nuisance_linear__candidates_ridge__overlap_06-224.parquet")            
            }, None, True, True, (-0.5, 1.05)
            ),
]
dataset_label = "Dataset"

@pytest.mark.parametrize(
    "xp_paths, reference_metric, plot_legend, plot_middle_bin, xlim",
    [
        *DATASET_EXPERIMENTS
    ],
)
def test_plot_effect_ratio_difference(
    xp_paths: Dict[str, Path],
    reference_metric: str,
    plot_legend: bool,
    plot_middle_bin: bool,
    xlim: Tuple[float, float],):
    
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
        kendall_by_effect_ratio_bin, evaluation_metric = get_kendall_by_effect_ratio_bin(
            xp_res=xp_res_,
            reference_metric=reference_metric,
            expe_causal_metrics=expe_causal_metrics,
            plot_middle_overlap_bin=plot_middle_bin
        )
        kendall_by_effect_ratio_bin[DATASET_LABEL] = expe_name
        all_expe_results_by_bin.append(kendall_by_effect_ratio_bin)
    all_expe_results_by_bin_df = pd.concat(all_expe_results_by_bin)
    ds_used = all_expe_results_by_bin_df[DATASET_LABEL].unique()
    ds_order = [ds_label_ for ds_label_ in DATASETS_PALETTE.keys() if ds_label_ in ds_used]
    # Weak overlap first
    overlap_order = [ov_ for ov_ in EFFECT_RATIO_BIN_LABELS if ov_ in all_expe_results_by_bin_df[EFFECT_RATIO_BIN_COL].unique()][::-1]
    metric_order = [CAUSAL_METRIC_LABELS[metric_label_] for metric_label_ in METRIC_ORDER]    

    # Full figure
    g = sns.FacetGrid(
        data=all_expe_results_by_bin_df, 
        col=DATASET_LABEL, 
        col_wrap=2, 
        col_order=ds_order, 
        height=10, aspect=1)
    g = g.map_dataframe(
        sns.boxplot,
        x=evaluation_metric,
        y=METRIC_LABEL,
        order=metric_order,
        hue=EFFECT_RATIO_BIN_COL,
        hue_order=overlap_order,
        palette=EFFECT_RATIO_BIN_PALETTE,
        linewidth=2,
        notch=True,
        medianprops={"linewidth": 6},

    )
    if f"{evaluation_metric}_long" in EVALUATION_METRIC_LABELS.keys():
        suptitle = EVALUATION_METRIC_LABELS[f"{evaluation_metric}_long"]
    else:
        suptitle = EVALUATION_METRIC_LABELS[evaluation_metric]
    g.set(xlabel=suptitle, ylabel="", xlim=xlim)
    if reference_metric is not None:
        ref_metric_str = f"_ref_metric_{reference_metric}"
    else:
        ref_metric_str = "kendall"
    if plot_legend:
        g.add_legend()
        legend_data = g._legend_data # type: ignore
        g._legend.remove()# type: ignore
        g.add_legend(
            title=EFFECT_RATIO_BIN_COL, 
            legend_data=legend_data,
        )        
    else:
        g._legend.remove()# type: ignore
    save_figure_to_folders(
        figure_name=Path(script_name+f"/effect_ratio_by_bin_comparaison_{ref_metric_str}_by_{DATASET_LABEL}"),
        figure_dir=True,
        notes_dir=False,
        paper_dir=True,
    )
    
    # R_risk only figure
    sub_metrics = [CAUSAL_METRIC_LABELS["r_risk"], CAUSAL_METRIC_LABELS["oracle_r_risk"]]
    r_risk_data = all_expe_results_by_bin_df[all_expe_results_by_bin_df[METRIC_LABEL].isin(sub_metrics)]
    g = sns.FacetGrid(
    data=r_risk_data, col=DATASET_LABEL, col_order=ds_order,height=6, aspect=1.8, col_wrap=2)
    g = g.map_dataframe(
        sns.boxplot,
        x=evaluation_metric,
        y=METRIC_LABEL,
        order=sub_metrics,
        hue=EFFECT_RATIO_BIN_COL,
        hue_order=overlap_order,
        palette=EFFECT_RATIO_BIN_PALETTE,
        linewidth=2,
        notch=True,
        medianprops={"linewidth": 6},
    )
    g.set(xlabel=suptitle, ylabel="", xlim=(0, 1.05))
    g.add_legend(
            legend_data=legend_data,
            loc="upper center",
            ncol=3, bbox_to_anchor=(0.35, 1.05),
            columnspacing=0.5,
            prop={"size": 36},
        )

    save_figure_to_folders(
        figure_name=Path(script_name+f"/effect_ratio_by_bin_comparaison_{ref_metric_str}_by_{DATASET_LABEL}_r_risk_only"),
        figure_dir=True,
        notes_dir=False,
        paper_dir=True,
    )
    
    return
    

def test_plot_metrics_legend():
    ncol=2
    plot_metric_legend(CAUSAL_METRICS,ncol=ncol)
    save_figure_to_folders(
        figure_name=Path(f"legend_metrics_ncol={ncol}"),
        figure_dir=True,
        notes_dir=False,
        paper_dir=True,
    )

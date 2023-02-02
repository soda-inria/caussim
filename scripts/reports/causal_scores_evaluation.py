import re
from typing import Tuple
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
    CAUSAL_METRICS,
    METRIC_OF_INTEREST_LABELS,
    plot_metric_legend,
    plot_ranking_aggregation,
)
from caussim.reports.utils import get_metric_rankings_by_dataset, get_rankings_aggregate
from caussim.utils import *
from caussim.config import *

sns.set_style("whitegrid")


PAPER_EXPERIENCES = [
    (Path(DIR2EXPES / "acic_2018_save" / "acic_2018__nuisance_non_linear__candidates_hist_gradient_boosting__first_uid_432"), False, 0.6, (1e-4, 1e0), (0.2, 1.05), None),
    (
        Path(
            DIR2EXPES
            / "caussim_save"
            / "caussim__stacked_regressor__test_size_5000__n_datasets_1000"
        ),
        False,
        0.6,
        (1e-2, 5 * 1e1),
        (0, 1.05),
        None,
    ),
    (
        Path(
            DIR2EXPES
            / "acic_2016_save"
            / "acic_2016__stacked_regressor__dgp_1-77__seed_1-10"
        ),
        False,
        0.6,
        (1e-5, 0.25),
        (0.5, 1.05),
        None,
    ),
    (
        Path(
            DIR2EXPES
            / "twins_save"
            / "twins__stacked_regressor__rs_1-10__overlap_0.1-3"
        ),
        False,
        0.6,
        (1e-5, 0.25),
        (0, 1.05),
        None,
    ),
]


mean_risk_xps_config = []
for xp_tupple in PAPER_EXPERIENCES:
    xp_tupple_ = list(xp_tupple)
    xp_tupple_[-1] = "mean_risks"
    xp_tupple_[-2] = (-1, 1)
    mean_risk_xps_config.append(tuple(xp_tupple_))
PAPER_EXPERIENCES = mean_risk_xps_config


@pytest.mark.parametrize(
    "xp_path, show_legend, quantile, ylim_bias_to_tau_risk, ylim_ranking, reference_metric",
    [
        *PAPER_EXPERIENCES
        # (Path(DIR2EXPES / "caussim_save"/
        # "caussim__linear_regressor__test_size_5000__n_datasets_1000"), True,
        # 0.6, (1e-2, 5e1), (0, 1.05), None), # Here, nuisances are learned with
        # ridge regressor, which yielsd poor performance of the selection
        # procedure.
        #(Path(DIR2EXPES / "caussim_save"/ "caussim__nuisance_non_linear__candidates_ridge__overlap_01-247_join_nuisance_train_set"), False, 0.6, (1e-2, 5*1e1), (0, 1.05), None), # a 
        # (Path(DIR2EXPES / "caussim_save"/ "caussim__nuisance_non_linear__candidates_ridge__overlap_01-247_join_nuisance_train_set"), False, 0.6, (1e-2, 5*1e1), (0, 1.05), None), # a balanced treatment ratio xp with 900 instances and 4 seeds; it shows that different seeds for response surfaces are crucial to draw robust conclusion.
        #separated train and nuisances sets
        # (
        #     Path(
        #         DIR2EXPES
        #         / "caussim_save"
        #         / "caussim__nuisance_non_linear__candidates_ridge__overlap_01-247_separated_nuisance_train_set"
        #     ),
        #     False,
        #     0.6,
        #     (1e-2, 5 * 1e1),
        #     (-1, 1.05),
        #     "mean_risks",
        # )
    ],
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
    HTY_PLOT = False
    RANKING_COMPUTE = True
    BEST_ESTIMATOR_PLOT = False

    expe_results, _ = read_logs(xp_path)
   
    dataset_name = expe_results["dataset_name"].values[0]
    # subsetting
    subset_of_xp = expe_results["test_d_normalized_tv"].value_counts().index[:100]
    expe_results = expe_results.loc[
        expe_results["test_d_normalized_tv"].isin(subset_of_xp)
    ]
    xp_causal_metrics = [
        metric for metric in CAUSAL_METRICS if metric in expe_results.columns
    ]

    if dataset_name == "acic_2018":
        xp_causal_metrics = [
            m for m in xp_causal_metrics if re.search("oracle|gold", m) is None
        ]

        overlap_measure = "hat_d_normalized_tv"
    else:
        overlap_measure = "test_d_normalized_tv"
    if dataset_name == "acic_2018":
        expe_results["test_d_normalized_tv"] = expe_results["hat_d_normalized_tv"]
    nuisance_models_label = get_nuisances_type(expe_results)
    # ### Exclcude some runs with extrem values ### #
    max_mse_ate = 1e5
    outliers_mask = expe_results["mse_ate"] >= max_mse_ate
    outliers = expe_results.loc[outliers_mask]
    expe_results = expe_results.loc[~outliers_mask]
    logging.warning(f"Removing {outliers_mask.sum()} with mse_ate>{max_mse_ate}")
    candidate_params = get_candidate_params(expe_results)
    overlap_measure, expe_indices = get_expe_indices(expe_results)
    
    if HTY_PLOT:
        expe_indices += [
            "log_treatment_heterogeneity_norm",
            "treatment_heterogeneity_norm",
            "heterogeneity_score_norm",
        ]

    logging.info(
        f"Nb (simulations x test_seed) : {len(expe_results[expe_indices].drop_duplicates())}"
    )
    if BEST_ESTIMATOR_PLOT:
        best_estimator_by_dataset = get_best_estimator_by_dataset(
            expe_logs=expe_results,
            expe_indices=expe_indices,
            candidate_params=candidate_params,
            causal_metrics=xp_causal_metrics,
        )

        comparison_df_w_random = pd.concat(
            [
                best_estimator_by_dataset[metric]
                for metric in [*xp_causal_metrics, "random"]
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
            overlap_measure,
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
        # evaluation_measures = []

        for eval_m in evaluation_measures:
            moi = overlap_measure
            ax = plot_evaluation_metric(
                comparison_df_w_best_as_oracle=comparison_df_w_best_as_oracle,
                evaluation_metric=eval_m,
                measure_of_interest_name=moi,
                selection_metrics=xp_causal_metrics,
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
        expe_rankings = get_metric_rankings_by_dataset(
            expe_results=expe_results,
            expe_indices=expe_indices,
            causal_metrics=xp_causal_metrics,
            candidate_params=candidate_params,
        )

        rankings_agg = get_rankings_aggregate(
            expe_rankings=expe_rankings,
            expe_indices=expe_indices,
            causal_metrics=xp_causal_metrics,
        )
        if HTY_PLOT:
            # Quantile direction cuts the datasets in several bins (eg. heterogeneity bins) and compute a ranking plot on each bin along the direction of interest (eg. overlap)
            quantiles_direction = "treatment_heterogeneity_norm"

            # Treatment heterogeneity sub-plots for terciles of overlap
            quantiles_ = np.quantile(
                expe_results[quantiles_direction].unique(), q=[0.33, 0.66]
            )
            eps = 1e-6
            bins_low = [np.min(expe_results[quantiles_direction]) - eps, *quantiles_]
            bins_sup = [*quantiles_, np.max(expe_results[quantiles_direction]) + eps]
            bins_ = [(b_low, b_sup) for (b_low, b_sup) in zip(bins_low, bins_sup)]
            # overlap_bins = [(0,1)]
            for i, (bin_min, bin_max) in enumerate(bins_):
                rankings_agg_by_bin = rankings_agg.loc[
                    (rankings_agg[quantiles_direction] >= bin_min)
                    & (rankings_agg[quantiles_direction] < bin_max)
                ]
                for metric_of_interest in [
                    overlap_measure
                ]:  # "treatment_heterogeneity_norm", ,"heterogeneity_score_norm"]:
                    ax = plot_ranking_aggregation(
                        rankings_aggregation=rankings_agg_by_bin,
                        expe_indices=expe_indices,
                        x_metric_name=metric_of_interest,
                        lowess_type="lowess_quantile",
                        lowess_kwargs={"frac": 0.66, "it": 10, "quantile": quantile},
                        show_legend=False,
                        y_lim=ylim_ranking,
                        reference_metric=None,
                    )
                    ax.text(
                        0,
                        0.92,
                        s=f"{METRIC_OF_INTEREST_LABELS[quantiles_direction]}: {bin_min:.2f}-{bin_max:.2f}",
                        transform=plt.gcf().transFigure,
                        fontsize=20,
                    )
                    xmin = np.percentile(rankings_agg_by_bin[metric_of_interest], 5)
                    xmax = np.percentile(rankings_agg_by_bin[metric_of_interest], 95)
                    ax.set(xlim=(np.max([xmin, -1.5]), xmax))
                    plt.tight_layout()
                    save_figure_to_folders(
                        figure_name=Path(
                            f"kendalls_tau_{metric_of_interest}/{xp_path.stem}_{quantiles_direction}_tercile_{i}"
                        ),
                        figure_dir=True,
                        notes_dir=True,
                        paper_dir=False,
                    )
        if reference_metric is not None:
            #ylim_ranking = (ylim_ranking[0] - ylim_ranking[1], ylim_ranking[1])
            ref_metric_str = f"_ref_metric_{reference_metric}"
        else:
            ref_metric_str = ""
        ax = plot_ranking_aggregation(
            rankings_aggregation=rankings_agg,
            expe_indices=expe_indices,
            x_metric_name=overlap_measure,
            lowess_type="lowess_quantile",
            lowess_kwargs={"frac": 0.66, "it": 10, "quantile": quantile},
            show_legend=False,
            y_lim=ylim_ranking,
            reference_metric=reference_metric,
        )
        plt.tight_layout()

        save_figure_to_folders(
            figure_name=Path(f"kendalls_tau{ref_metric_str}/{xp_path.stem}"),
            figure_dir=True,
            notes_dir=False,
            paper_dir=True,
        )


def test_plot_metrics_legend():
    plot_metric_legend(CAUSAL_METRICS)
    save_figure_to_folders(
        figure_name=Path(f"legend_metrics"),
        figure_dir=False,
        notes_dir=False,
        paper_dir=True,
    )

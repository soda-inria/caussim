from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
from pyrsistent import l
import pytest
from caussim.config import DIR2EXPES, DIR2FIGURES_T, DIR2PAPER_IMG
from caussim.reports import read_logs, CAUSAL_METRICS,  plot_agreement_w_tau_risk, get_best_estimator_by_dataset
from caussim.reports.plots_utils import plot_evaluation_metric, plot_metric_legend
from caussim.reports.utils import get_metric_rankings_by_dataset, get_nuisances_type, get_rankings_aggregate

xp_results, _ = read_logs(Path(DIR2EXPES / "acic_2016_save" / "acic_2016__stacked_regressor__dgp_1-77__seed_1-10"))
xp_causal_metrics = [
    metric for metric in CAUSAL_METRICS if metric in xp_results.columns
]
overlap_measure = "test_d_normalized_tv"
expe_indices = [
    "simulation_seed",
    "simulation_param",
    overlap_measure,
]
candidate_params = [
    "final_estimator__learning_rate",
    "meta_learner_name",
    "final_estimator__max_leaf_nodes",
]
n_datasets =  len(xp_results[expe_indices].drop_duplicates())
nuisance_models_label = get_nuisances_type(xp_results)

@pytest.fixture(scope="module")
def best_estimator_by_dataset():
    best_estimator_by_dataset = get_best_estimator_by_dataset(
        expe_logs=xp_results,
        expe_indices=expe_indices, 
        candidate_params=candidate_params,
        causal_metrics=xp_causal_metrics
    )
    return best_estimator_by_dataset

def test_get_best_estimator_by_dataset(best_estimator_by_dataset):
    assert set(xp_causal_metrics+["random"]) == set(best_estimator_by_dataset.keys())
    for best_estimator_by_method in best_estimator_by_dataset.values():
        assert len(best_estimator_by_method) == n_datasets


def test_plot_agreement_with_tau_risk(best_estimator_by_dataset):
    comparison_df_w_random = pd.concat(
        [best_estimator_by_dataset[metric] for metric in [*xp_causal_metrics, "random"]]
    )
    comparison_df_w_best_as_oracle = comparison_df_w_random.merge(
        best_estimator_by_dataset["tau_risk"][["tau_risk", "run_id", *expe_indices]],
        on=expe_indices,
        how="left",
        suffixes=("", "_as_oracle"),
    )

    fig, _ = plot_agreement_w_tau_risk(
        comparison_df_w_best_as_oracle,
        overlap_measure,
        nuisance_models_label="Non Linear",
        show_legend=False
    )
    plt.tight_layout()
    dir2save  = DIR2FIGURES_T / "agreement_w_tau_risk"
    (dir2save).mkdir(exist_ok=True, parents=True)
    plt.savefig(str(dir2save /"acic_2016__stacked_regressor__dgp_1-77__seed_1-10.pdf"), bbox_inches="tight")
    

def test_plot_evaluation_metric(best_estimator_by_dataset):
    comparison_df_w_random = pd.concat(
        [best_estimator_by_dataset[metric] for metric in [*xp_causal_metrics, "random"]]
    )
    comparison_df_w_best_as_oracle = comparison_df_w_random.merge(
        best_estimator_by_dataset["tau_risk"][["tau_risk", "run_id", *expe_indices]],
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
    eval_m = "normalized_bias_tau_risk_to_best_method"
    plot_evaluation_metric(
        comparison_df_w_best_as_oracle=comparison_df_w_best_as_oracle,
        evaluation_metric=eval_m,
        measure_of_interest_name=overlap_measure,
        selection_metrics=xp_causal_metrics,
        lowess_type="lowess_quantile",
        lowess_kwargs={"frac": 0.6, "num_fits": 10, "quantile": 0.6},
        nuisance_models_label=nuisance_models_label,
        ylog=True,
        linewidth=6
    )
    plt.tight_layout()
    dir2save = DIR2FIGURES_T / eval_m 
    dir2save.mkdir(exist_ok=True, parents=True)
    plt.savefig(str(dir2save / "acic_2016__stacked_regressor__dgp_1-77__seed_1-10.pdf"), bbox_inches="tight")
    
    
def test_plot_legend():
    ax = plot_metric_legend(xp_causal_metrics)
    plt.savefig(DIR2PAPER_IMG / "legend_metrics.pdf", bbox_inches="tight")
    

@pytest.fixture()
def metric_rankings_by_dataset():
    selected_xp_results = xp_results.loc[xp_results["simulation_param"].isin([5, 6]) & (xp_results["simulation_seed"]==1)]
    
    rankings = get_metric_rankings_by_dataset(
        expe_results=selected_xp_results, 
        expe_indices=expe_indices,
        causal_metrics=xp_causal_metrics,
        candidate_params=candidate_params
        )
    return rankings

def test_get_metric_rankings_by_dataset(metric_rankings_by_dataset):
    expected_ranking_mu_risk = np.array([17, 16, 15, 14, 13, 12, 0,2 , 1, 3, 4, 5, 11, 9, 6, 7, 10, 8])
    assert np.array_equal(
        metric_rankings_by_dataset[:18]["ranking_mu_risk"],
        expected_ranking_mu_risk
    )
    assert metric_rankings_by_dataset.shape[0] == 2*18

def test_get_rankings_aggregate(metric_rankings_by_dataset):
    ranking_agg = get_rankings_aggregate(
        expe_rankings=metric_rankings_by_dataset,
        expe_indices=expe_indices,
        causal_metrics=xp_causal_metrics
    )
    expected_kendall_mu_risk = np.array([0.908, 0.856])
    assert np.array_equal(
        np.round(ranking_agg["kendalltau_stats__tau_risk_mu_risk"], 3),
        expected_kendall_mu_risk
    )
    assert ranking_agg.shape[0] == 2
    
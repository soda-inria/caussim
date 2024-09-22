#
import re
from typing import Tuple
from matplotlib.patches import Patch
import pandas as pd
import numpy as np

import pytest
from caussim.reports import (
    read_logs,
    save_figure_to_folders,
)
from caussim.reports.plots_utils import (
    CAUSAL_METRICS, EVALUATION_METRIC_LABELS, OVERLAP_BIN_COL, OVERLAP_BIN_LABELS, CAUSAL_METRIC_LABELS)
from caussim.reports.utils import get_best_estimator_by_dataset, get_candidate_params, get_expe_indices
from caussim.utils import *
from caussim.config import *

sns.set_style("whitegrid")

dataset_label = "Dataset"
TRAIN_SIZE_LABEL = "Train ratio"
# label stuff
train_size_33_label = "0.33" 
train_size_50_label = "0.5" 
train_size_66_label = "0.66" 
train_size_90_label = "0.9"
train_size_80_label = "0.8"
train_size_95_label = "0.95"
train_size_order = [
    train_size_33_label,
    train_size_50_label, 
    train_size_66_label,
    train_size_80_label,
    train_size_90_label,
    train_size_95_label
    ]

TRAIN_SIZE_PALETTE =  dict(zip(
    train_size_order, sns.color_palette("viridis", n_colors=len(train_size_order)))
                           )

train_size_datasets = {
            train_size_33_label: Path(DIR2EXPES / "acic_2016_save"/ "acic_2016__nuisance_stacking__candidates_hist_gradient_boosting__vset_30__tset_33__dgp_1-77__rs_1-10.parquet"),
            train_size_50_label: Path(DIR2EXPES / "acic_2016_save"/ "acic_2016__nuisance_stacking__candidates_hist_gradient_boosting__vset_30__tset_50__dgp_1-77__rs_1-10.parquet"),
            train_size_66_label: Path(DIR2EXPES / "acic_2016_save"/ "acic_2016__nuisance_stacking__candidates_hist_gradient_boosting__vset_30__tset_66__dgp_1-77__rs_1-10.parquet"),
            train_size_80_label:Path(
                DIR2EXPES
                / "acic_2016_save"
                / "acic_2016__nuisance_stacking__candidates_hist_gradient_boosting__vset_30__tset_80__dgp_1-77__rs_1-10.parquet"),
            train_size_90_label:Path(
                DIR2EXPES
                / "acic_2016_save"
                / "acic_2016__nuisance_stacking__candidates_hist_gradient_boosting__vset_30__tset_90__dgp_1-77__rs_1-10.parquet"),
            train_size_95_label:Path(
                DIR2EXPES
                / "acic_2016_save"
                / "acic_2016__nuisance_stacking__candidates_hist_gradient_boosting__vset_30__tset_95__dgp_1-77__rs_1-10.parquet"),
            }

TRAIN_SIZE_EXPERIMENTS = [
            (train_size_datasets, "r_risk", "validation_tau_risk",  False, False,(0, 10), True),
            (train_size_datasets, "r_risk", "validation_test_abs_bias_ate", True, False,(1.5, 2.5), False),
            #(train_size_datasets, "r_risk", "validation_abs_bias_ate",  True, (-0.05, 1)),
            #(train_size_datasets, "r_risk", "validation_abs_bias_ate",  True,
            #(-0.05, 0.4))
            #(train_size_datasets, "r_risk", "validation_abs_bias_ate",  True, False, (-0.05, 0.4)),
            #(train_size_datasets, "r_risk", "validation_test_abs_bias_ate", False, False, None),#(-0.05, 1)),
            #(train_size_datasets, "r_risk", "validation_test_abs_bias_ate", True, False, (1.5, 2.5)),#(-0.05, 1)),
            #(train_size_datasets, "r_risk", "abs_bias_ate", True, False, (0, 0.25), False),
]
evaluation_metrics_normalisation = {
    'validation_tau_risk': '',
    'validation_abs_bias_ate': 'validation_ate',
    'validation_test_abs_bias_ate': 'validation_ate',
    "abs_bias_ate": "ate"
}

def get_evaluation_metric_str_train_test_split(evaluated_metric: str, evaluation_metric: str, relative_to_best_tau_model: bool) -> str:
    if evaluation_metric == "validation_tau_risk":
        label = r"$\widehat{\tau_{\mathcal{V}}\mathrm{-risk}}(f^*($" + CAUSAL_METRIC_LABELS[evaluated_metric] + r"$))$"
    elif evaluation_metric == "validation_abs_bias_ate":
        label = r"$|\hat \tau_{\mathcal{V}} - \hat \tau_{\mathcal{V}}(f^*($" + CAUSAL_METRIC_LABELS[evaluated_metric] + r"$)|$"
    elif evaluation_metric == "validation_test_abs_bias_ate":
        label = r"$|\tau_{\mathcal{V}} - \hat \tau_{\mathcal{S}}(f^*($" + CAUSAL_METRIC_LABELS[evaluated_metric] + r"$))|$"
    elif evaluation_metric == "abs_bias_ate":
        label =  r"$|\tau_{\mathcal{S}} - \hat \tau_{\mathcal{S}}(f^*($" + CAUSAL_METRIC_LABELS[evaluated_metric] + r"$))|$"
    else:
        raise ValueError(f"evaluation_metric should be in {evaluation_metrics_normalisation.keys()}, got {evaluated_metric}")
    if relative_to_best_tau_model:
        label+= f"relative to best model selected by {CAUSAL_METRIC_LABELS['tau_risk']}"
    return label

@pytest.mark.parametrize(
    "train_size_datasets, evaluated_metric, evaluation_metric, normalization, best_tau_normalization, xlim, plot_legend",
    [*TRAIN_SIZE_EXPERIMENTS]
)
def test_ranking_r_risk_only(
    train_size_datasets: Dict[str, str],
    evaluated_metric: str,
    evaluation_metric: str,
    normalization: bool,
    best_tau_normalization: bool, 
    plot_legend:bool,
    xlim,
):
    """_summary_

    Parameters
    ----------
    train_size_datasets : Dict[str, str]
        The datasets with different training set sizes
    evaluated_metric : str
        The causal metric to be plot: mostly r_risk but i could look at other metrics
    evaluation_metric : str
        The evaluation metric: bias ate, PEHE (tau-risk) focused on cate
    normalization : bool
        normalization by an oracle, defined by the
        evaluation_metrics_normalisation dict
    best_tau_normalization : bool
        normalize compared to te best tau risk model selected by the oracle tau-risk
    plot_legend : bool
        _description_
    xlim : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    
    plot_middle_overlap_bin = True
    all_expe_results_by_bin = []
    for procedure_type, xp_path in train_size_datasets.items():
        xp_res_, _ = read_logs(xp_path)
        expe_causal_metrics = [
            metric for metric in CAUSAL_METRICS if ((metric in xp_res_.columns) and (metric not in ["mu_risk", "mu_iptw_risk", "oracle_mu_iptw_risk","oracle_u_risk", "u_risk", "w_risk", "oracle_w_risk"]))
        ]
        overlap_measure, expe_indices = get_expe_indices(xp_result=xp_res_)
        candidate_params = get_candidate_params(xp_result=xp_res_)
        # fix because validation ate was not in the original experiments. I
        # computed a dummy xp whith the same random seeds, and the same
        # validation set. The following code add the column. 
        if "validation_ate" not in xp_res_.columns:
            validation_ate_df = pd.read_parquet(DIR2EXPES/"acic_2016_save"/"validation_ate_df__vset_30_vseed_1_tset50__dgp_1-77__rs_1-10.parquet")
            xp_res_ = xp_res_.merge(
             validation_ate_df[["seed","dgp","validation_ate"]], on=["seed","dgp"], how="left", validate="many_to_one"
            )
        # add the bias between the validation ate and the test predicted ate
        xp_res_['validation_test_abs_bias_ate'] = np.abs(xp_res_['bias_ate'] - + xp_res_['ate'] - xp_res_['validation_ate'])
        best_estimator_by_dataset = get_best_estimator_by_dataset(
            expe_logs=xp_res_,
            expe_indices=expe_indices,
            candidate_params=candidate_params,
            causal_metrics=expe_causal_metrics,
        )
        evaluation_metric_label = get_evaluation_metric_str_train_test_split(evaluated_metric=evaluated_metric, evaluation_metric=evaluation_metric, relative_to_best_tau_model=best_tau_normalization)
        
        evaluated_metric_best = best_estimator_by_dataset[evaluated_metric]
        if normalization:
            norm_metric = evaluation_metrics_normalisation[evaluation_metric]
            evaluation_metric_label = evaluation_metric_label + f" normalized by {EVALUATION_METRIC_LABELS[norm_metric]}"
            evaluated_metric_best[evaluation_metric_label] = evaluated_metric_best[evaluation_metric] / np.abs(evaluated_metric_best[norm_metric])
            
        else:
            evaluated_metric_best[evaluation_metric_label] = evaluated_metric_best[evaluation_metric]
        
        if best_tau_normalization:
            tau_risk_best = best_estimator_by_dataset["tau_risk"]
            evaluated_metric_best[evaluation_metric_label] = (evaluated_metric_best[evaluation_metric_label] - tau_risk_best[evaluation_metric])/ tau_risk_best[evaluation_metric]
        evaluated_metric_best[TRAIN_SIZE_LABEL] =  procedure_type
        evaluated_metric_best[dataset_label] = xp_res_["dataset_name"].values[0].replace("_", " ").capitalize()
        # groupby overlap bin
        bins_quantiles = [0,0.33, 0.66, 1]
        bins_values = evaluated_metric_best[overlap_measure].quantile(bins_quantiles).values
        evaluated_metric_best[OVERLAP_BIN_COL] = pd.cut(evaluated_metric_best[overlap_measure], bins=bins_values, labels=OVERLAP_BIN_LABELS).astype(str)
        # keep only extrem tertiles
        if plot_middle_overlap_bin:
            kept_bins = [OVERLAP_BIN_LABELS[0], OVERLAP_BIN_LABELS[1], OVERLAP_BIN_LABELS[2]]
        else:
            kept_bins = [OVERLAP_BIN_LABELS[0], OVERLAP_BIN_LABELS[2]]
        evaluated_metric_best = evaluated_metric_best.loc[evaluated_metric_best[OVERLAP_BIN_COL].isin(kept_bins)] 
        
        all_expe_results_by_bin.append(evaluated_metric_best)
    all_expe_results_by_bin_df = pd.concat(all_expe_results_by_bin)
    hue_order=train_size_order
    all_expe_results_by_bin_df["causal_metric"] = all_expe_results_by_bin_df["causal_metric"].map(lambda x: CAUSAL_METRIC_LABELS[x])
    aspect=1.8
    g = sns.FacetGrid(data=all_expe_results_by_bin_df, col=OVERLAP_BIN_COL,
        col_order=[OVERLAP_BIN_LABELS[0], OVERLAP_BIN_LABELS[2]], aspect=aspect,
        height=6)
    g = g.map_dataframe(
        sns.boxplot,
        x=evaluation_metric_label,
        y="causal_metric",
        hue=TRAIN_SIZE_LABEL,
        hue_order=hue_order,
        notch=True,
        palette=TRAIN_SIZE_PALETTE,
    )
    g.set_titles(col_template="{col_name}")
    g.set(xlabel="", ylabel="", xlim=xlim)
    g.fig.text(x=0.3, y=0.1, s=evaluation_metric_label)#,ha="center", va="center")
    # prune last xtick label
    for a in g.axes.flat:
        # get all the labels of this axis
        labels = a.get_xticklabels()
        # remove the first and the last labels
        labels[0] = labels[-1] = ""
        # set these new labels
        a.set_xticklabels(labels)
        a.set_yticklabels([""])

    if xlim is None:
        xlim = None
    if plot_legend:
        #g.add_legend(title=TRAIN_SIZE_LABEL, loc="upper right")
        legend_handles = [
        plt.Line2D([0], [0], color="white")] + [
        Patch(facecolor=facecolor_, edgecolor="black") for facecolor_ in TRAIN_SIZE_PALETTE.values()
        ]
        legend_labels = [TRAIN_SIZE_LABEL, *list(TRAIN_SIZE_PALETTE.keys())]
        g.add_legend(
            title="", 
            handles=legend_handles, 
            labels=legend_labels, 
            loc="lower center", ncol=len(TRAIN_SIZE_PALETTE)+1, bbox_to_anchor=(0.35,0.95),
            columnspacing=0.5
            )
    
    save_figure_to_folders(
        figure_name=Path(f"_5_train_size/evaluated_metric_{evaluated_metric}__evaluation_{evaluation_metric}__best_tau_norm__{best_tau_normalization}_norm_{normalization}__acic16"),
        figure_dir=True,
        notes_dir=False,
        paper_dir=True,
    )
    return g
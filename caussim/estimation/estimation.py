from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.utils import check_consistent_length
from sklearn.utils.estimator_checks import check_is_fitted

from caussim.config import COLOR_MAPPING, LABEL_MAPPING
from caussim.data.causal_df import CausalDf
from caussim.estimation.estimators import (
    AteEstimator,
    CateEstimator,
    get_meta_learner_components,
)
from caussim.estimation.scores import brier_skill_score, ipw_risk, r_risk, u_risk, w_risk
from caussim.pdistances.mmd import normalized_total_variation, total_variation_distance


class CaussimLogger(object):
    """
    A class to log individual results for causal inference simulation.

    Parameters
    ----------
    object : _type_
        _description_
    """

    def __init__(self) -> None:
        self.results = []

    def log_simulation_result(
        self,
        causal_df: CausalDf,
        causal_estimator: Union[AteEstimator, CateEstimator],
        estimates: Dict[str, float],
        parameters: Dict[str, float],
    ):
        if type(causal_estimator) == AteEstimator:
            simu_results = self.log_simulation_result_ate(
                causal_df=causal_df,
                causal_estimator=causal_estimator,
                estimates=estimates,
            )

            simu_results.update(parameters)
            self.results.append(simu_results)
        else:
            raise NotImplementedError

    def log_simulation_result_ate(
        self,
        causal_df: CausalDf,
        causal_estimator: AteEstimator,
        estimates: Dict[str, float],
    ) -> None:
        simu_results = {}
        # dataset info
        simu_results["dataset_name"] = causal_df.dataset_name
        simu_results["dgp"] = causal_df.dgp
        simu_results["overlap"] = causal_df.overlap_parameter
        simu_results["random_state"] = causal_df.random_state
        simu_results.update(causal_df.describe())
        # model info
        simu_results["meta_learner"] = causal_estimator.meta_learner_type
        simu_results.update(causal_estimator.metrics)
        simu_results["outcome_model"] = type(causal_estimator.outcome_model).__name__
        simu_results["propensity_model"] = type(
            causal_estimator.propensity_model
        ).__name__
        simu_results.update(
            {
                f"y_model{k}": v
                for (k, v) in causal_estimator.outcome_model.get_params().items()
            }
        )
        simu_results.update(
            {
                f"a_model{k}": v
                for (k, v) in causal_estimator.propensity_model.get_params().items()
            }
        )

        # oracle and feasible estimates info
        oracle_results = causal_df.estimate_oracles()
        simu_results["ate"] = oracle_results["ate"]
        simu_results["hat_ate"] = estimates["hat_ate"]
        simu_results["treatment_heterogeneity"] = oracle_results[
            "treatment_heterogeneity"
        ]
        return simu_results

    def to_pandas(self):
        return pd.DataFrame(self.results)


# ### Scoring for selection metrics ### #
def get_selection_metrics(causal_df: CausalDf, predictions: pd.DataFrame) -> Dict:
    simulation_scores = {}
    # oracles
    df = causal_df.df
    test_oracles = causal_df.estimate_oracles()
    simulation_scores["treatment_heterogeneity"] = test_oracles[
        "treatment_heterogeneity"
    ]
    simulation_scores["treatment_heterogeneity_obs"] = test_oracles[
        "treatment_heterogeneity_obs"
    ]
    ## effect ratio 
    simulation_scores["effect_ratio"] = test_oracles["effect_ratio"]
    ### First xps has a typo...
    simulation_scores["heterogeneity_score"] = test_oracles.get(
        "heterogeneity_score", test_oracles.get("hetereogeneity_score", None)
    )
    simulation_scores["check_e_min"] = predictions["check_e"].min()
    simulation_scores["check_e_max"] = predictions["check_e"].max()

    simulation_scores["ate"] = test_oracles["ate"]
    simulation_scores["bias_ate"] = predictions["hat_tau"].mean() - test_oracles["ate"]
    simulation_scores["abs_bias_ate"] = np.abs(simulation_scores["bias_ate"])
    simulation_scores["mse_ate"] = (
        predictions["hat_tau"].mean() - test_oracles["ate"]
    ) ** 2

    # Oracle
    simulation_scores["tau_risk"] = mean_squared_error(
        test_oracles["cate"], predictions["hat_tau"], squared=True
    )
    # reweighted surfaces differences
    simulation_scores["reweighted_surfaces_diff"] = np.mean(
        np.abs(df["mu_1"] - df["mu_0"]) * np.abs(2 * df["e"] - 1)
    )

    y = df["y"]
    a = df["a"]
    e = df["e"]
    # Directly from the R-decomposition
    oracle_m = df["mu_0"] * (1 - df["e"]) + df["mu_1"] * df["e"]
    hat_mu_1 = predictions["hat_mu_1"]
    hat_mu_0 = predictions["hat_mu_0"]
    hat_tau = predictions["hat_tau"]
    hat_y = hat_mu_1 * a + hat_mu_0 * (1 - a)
    check_e = predictions["check_e"]
    check_m = predictions["check_m"]
    # Feasibles
    ## Simple mse
    simulation_scores["r2"] = r2_score(y, hat_y)
    simulation_scores["mu_risk"] = mean_squared_error(y, hat_y, squared=True)
    
    ## ipws    
    check_ipw_weights = a /check_e + (1 - a) / (1 - check_e)
    simulation_scores["IPW_max"] = np.max(check_ipw_weights)
    simulation_scores["mu_iptw_risk"] = ipw_risk(y=y, a=a, hat_y=hat_y, hat_e=check_e)
    oracle_ipw_weights = a / e + (1 - a) / (1 - e)
    simulation_scores["oracle_IPW_max"] = np.max(oracle_ipw_weights)
    simulation_scores["oracle_mu_iptw_risk"] = ipw_risk(y=y, a=a, hat_y=hat_y, hat_e=e)
    ### IPW-N : population_thres
    check_e_probabilities = check_e * a + (1 - check_e) * (1 - a)
    check_e_probabilities_n = np.clip(check_e_probabilities, 1 / len(hat_y), 1)
    check_e_weigths_n = 1 / check_e_probabilities_n
    simulation_scores["ipw_n_max"] = np.max(check_e_weigths_n)
    simulation_scores["mu_ipw_n_risk"] = ipw_risk(y=y,a=a, hat_y=hat_y, hat_e=check_e_weigths_n)

    # R-risks
    simulation_scores["oracle_r_risk"] = r_risk(y=y,a=a, hat_m=oracle_m, hat_e=e, hat_tau=hat_tau)
    simulation_scores["oracle_r_risk_rewritten"] = np.sum(
        e * (1 - e) * (test_oracles["cate"] - hat_tau) ** 2
    ) / len(hat_tau)
    simulation_scores["r_risk"] = r_risk(y=y,a=a,hat_m=check_m, hat_e=check_e,hat_tau=hat_tau)
    simulation_scores["r_risk_gold_e"] = r_risk(y=y,a=a,hat_m=check_m, hat_e=e,hat_tau=hat_tau)
    simulation_scores["r_risk_gold_m"] = r_risk(y=y,a=a,hat_m=oracle_m, hat_e=check_e,hat_tau=hat_tau)

    # u_risk
    simulation_scores["u_risk"] = u_risk(y=y,a=a,hat_m=check_m, hat_e=check_e,hat_tau=hat_tau)
    simulation_scores["oracle_u_risk"] = u_risk(y=y,a=a,hat_m=oracle_m, hat_e=e,hat_tau=hat_tau)
    
    # w_risk 
    simulation_scores["w_risk"] = w_risk(y=y,a=a, hat_e=check_e,hat_tau=hat_tau)
    simulation_scores["oracle_w_risk"] = w_risk(y=y,a=a, hat_e=e,hat_tau=hat_tau)
    
    
    # Metrics for nuisances
    simulation_scores["check_e_bss"] = brier_skill_score(
        df["a"], predictions["check_e"]
    )
    simulation_scores["check_e_auroc"] = roc_auc_score(df["a"], predictions["check_e"])
    simulation_scores["check_e_mse"] = mean_squared_error(
        df["e"], predictions["check_e"]
    )
    simulation_scores["check_e_inv_mse"] = mean_squared_error(
        1 / df["e"], 1 / predictions["check_e"]
    )
    simulation_scores["check_e_IPW_mse"] = mean_squared_error(
        oracle_ipw_weights, check_ipw_weights
    )

    simulation_scores["check_m_r2"] = r2_score(df["y"], predictions["check_m"])
    simulation_scores["check_m_mse"] = mean_squared_error(
        df["y"], predictions["check_m"], squared=True
    )
    # Approximation of the normalized total variation
    simulation_scores["hat_d_normalized_tv"] = total_variation_distance(
        predictions["check_e"] / df["a"].mean(),
        (1 - predictions["check_e"]) / (1 - df["a"].mean()),
    )
    # decomposition of the mu_risks on each population
    mask_1 = df.query("a==1").index
    mask_0 = df.query("a==0").index
    simulation_scores["mu_risk_1"] = mean_squared_error(
        df.loc[mask_1, "y"], hat_y[mask_1]
    )
    simulation_scores["mu_risk_0"] = mean_squared_error(
        df.loc[mask_0, "y"], hat_y[mask_0]
    )
    simulation_scores["mu_iptw_risk_0"] = np.sum(
        ((df.loc[mask_0, "y"] - hat_y[mask_0]) ** 2) * oracle_ipw_weights[mask_0],
        axis=0,
    ) / len(hat_y[mask_0])
    simulation_scores["mu_iptw_risk_1"] = np.sum(
        ((df.loc[mask_1, "y"] - hat_y[mask_1]) ** 2) * oracle_ipw_weights[mask_1],
        axis=0,
    ) / len(hat_y[mask_1])

    return simulation_scores


# ### Utils for cate experiments : centered on nystroem and ridge configs ### #
def get_estimator_state(estimator: CateEstimator):
    # logging
    check_is_fitted(estimator)
    estimator_state = {}
    components = get_meta_learner_components(estimator.meta_learner_)
    for k, v in components.items():
        estimator_state[k] = v
    estimator_state["meta_learner_name"] = estimator.meta_learner.__class__.__name__
    estimator_state["meta_learner_params"] = str(estimator.meta_learner_.get_params())
    return estimator_state


def improved_HT(probabilities: np.array):
    """
    Implementation of improved Horvitz Thompson sampling

    References
    -----
    - (Zong et al., 2018) https://arxiv.org/pdf/1804.04255.pdf

    """

    # What is the purpose of K here ?
    sort_ix, sorted_probas = zip(*sorted(enumerate(probabilities), key=lambda x: x[1]))
    sorted_probas = np.array(sorted_probas)
    K = 0
    for i, p in enumerate(sorted_probas):
        if p <= (1 / (i + 2)):
            sorted_probas[:i] = 1 / (i + 2)
        K += 1
    return sorted_probas[np.argsort(np.array(sort_ix))]


def extrapolation_plot_residuals(
    causal_df: pd.DataFrame,
    hat_mu_1: np.array,
    hat_mu_0: np.array,
    fig=None,
    estimated_ps: np.array = None,
    loss: str = "l2",
    n_bins: int = 10,
    observed_outcomes_only: bool = False,
    show_sem: bool = True,
    normalize_y: bool = False,
):
    """Helper around extrapolation plot to plot residuals instead of outcomes.

    Parameters
    ----------
    causal_df : CausalDf
        _description_
    hat_y1 : np.array
        _description_
    hat_y0 : np.array
        _description_
    estimated_ps : np.array, optional
        _description_, by default None
    loss : str, optional
        _description_, by default "l2"
    n_bins : int, optional
        _description_, by default 10
    observed_outcomes_only : bool, optional
        _description_, by default False
    show_sem : bool, optional
        _description_, by default True
    normalize_y : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    supported_loss_ = ["l2", "l1"]
    if loss == "l2":
        loss_ = np.square
    elif loss == "l1":
        loss_ = np.abs
    else:
        raise ValueError(f"Support only losses in {supported_loss_}, got {loss}")
    if estimated_ps is None:
        ps = causal_df["e"]
    else:
        ps = estimated_ps
    residuals_1 = loss_(hat_mu_1 - causal_df["y_1"])
    residuals_0 = loss_(hat_mu_0 - causal_df["y_0"])
    ax, [ax_hist_0, ax_hist_1] = extrapolation_plot(
        y_0=residuals_0,
        y_1=residuals_1,
        ps=ps,
        a=causal_df["a"],
        fig=fig,
        n_bins=n_bins,
        observed_outcomes_only=observed_outcomes_only,
        show_sem=show_sem,
        normalize_y=normalize_y,
    )
    observed_only_label = " observed " if observed_outcomes_only else " "
    y_label = f"Mean{observed_only_label}{loss} residuals"
    if normalize_y:
        y_label += "\n(standardized)"
    ax.set(ylabel=y_label)
    return ax, [ax_hist_0, ax_hist_1]


def extrapolation_plot_outcome(
    causal_df: pd.DataFrame,
    estimated_ps: np.array = None,
    n_bins: int = 10,
    fig=None,
    observed_outcomes_only: bool = False,
    show_sem: bool = True,
    normalize_y: bool = False,
) -> Tuple[plt.Axes, Tuple[plt.Axes, plt.Axes]]:
    if estimated_ps is None:
        ps = causal_df["e"]
    else:
        ps = estimated_ps
    ax, [ax_hist_0, ax_hist_1] = extrapolation_plot(
        y_0=causal_df["y_0"],
        y_1=causal_df["y_1"],
        ps=ps,
        a=causal_df["a"],
        fig=fig,
        n_bins=n_bins,
        observed_outcomes_only=observed_outcomes_only,
        show_sem=show_sem,
        normalize_y=normalize_y,
    )
    if estimated_ps is not None:
        ax_hist_0.set(xlabel="Propensity score")
    return ax, [ax_hist_0, ax_hist_1]


def extrapolation_plot(
    y_0: np.array,
    y_1: np.array,
    ps: np.array,
    a: np.array,
    fig=None,
    n_bins: int = 10,
    observed_outcomes_only: bool = False,
    show_sem: bool = True,
    normalize_y: bool = False,
):
    """From a full CausalDf, plot the outcome values by bin f propensity score.

    Parameters
    ----------
    y_0 : np.array
        Control population outcomes or error values
    y_1 : np.array
        Treatment population outcomes or error values
    ps : np.array
        Probability scores, either oracle (for simulations) or estimated
    a : np.array
        Treatment status
    n_bins : int, optional
        Number of propensity score bins, by default 10
    observed_outcomes_only : bool, optional
        If showing only observed outcomes or both factual and counterfactucal ones, by default False
    show_sem : bool, optional
        If True show the standard error to the mean as error bar, by default True
    normalize_y: bool, optional
        If True, z-normalize the outcomes
    Returns
    -------
    _type_
        _description_
    """
    y_0_ = y_0.copy()
    y_1_ = y_1.copy()

    if observed_outcomes_only:
        y_0_.loc[a == 1] = np.nan
        y_1_.loc[a == 0] = np.nan
    if normalize_y:
        outcome_mean_ = np.nanmean(np.hstack([y_0_, y_1_]))
        outcome_sd_ = np.nanstd(np.hstack([y_0_, y_1_]))
        y_0_ = (y_0_ - outcome_mean_) / outcome_sd_
        y_1_ = (y_1_ - outcome_mean_) / outcome_sd_
    (
        bins_mean_outcomes,
        bins_sem_outcomes,
        bin_counts,
        bins_mean_ps,
    ) = get_mean_outcomes_by_ps_bin(
        y_0=y_0_,
        y_1=y_1_,
        ps=ps,
        n_bins=n_bins,
    )
    if fig is None:
        fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=(7, 2), bottom=0.1, top=0.9, hspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_histx_0 = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_histx_1 = ax_histx_0.twinx()
    # we declare statistical significance if the sem do not overlap
    alpha_non_significant = 0.3
    significance = (
        (bins_mean_outcomes[:, 0] + bins_sem_outcomes[:, 0])
        >= (bins_mean_outcomes[:, 1] - bins_sem_outcomes[:, 1])
    ) & (
        (bins_mean_outcomes[:, 0] - bins_sem_outcomes[:, 0])
        <= (bins_mean_outcomes[:, 1] + bins_sem_outcomes[:, 1])
    )

    alpha_ = alpha_non_significant * significance + (1 - significance)
    marker_size_ = 30
    for tt, ax_histx_i in zip([0, 1], [ax_histx_0, ax_histx_1]):
        if show_sem:
            for i in range(n_bins):
                ax.errorbar(
                    x=bins_mean_ps[i, tt],
                    y=bins_mean_outcomes[i, tt],
                    marker="o",
                    yerr=bins_sem_outcomes[i, tt],
                    color=COLOR_MAPPING[tt],
                    ms=marker_size_,
                    capsize=20,
                    elinewidth=3,
                    alpha=alpha_[i],
                )
        else:
            ax.scatter(
                x=bins_mean_ps[:, tt],
                y=bins_mean_outcomes[:, tt],
                color=COLOR_MAPPING[tt],
                s=marker_size_,
            )

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        sns.histplot(
            ax=ax_histx_i,
            x=ps[a == tt],
            color=COLOR_MAPPING[tt],
            alpha=0.6,
            fill=True,
            legend=False,
            bins=bins,
        )
        max_c = pd.cut(ps[a == tt], bins=bins).value_counts().max()
        ax_histx_i.set_ylabel(f"{LABEL_MAPPING[tt]}\ncount", color=COLOR_MAPPING[tt])
        ax_histx_i.set_yticks([max_c])
        ax_histx_i.set_yticklabels([max_c], color=COLOR_MAPPING[tt])
        ax_histx_i.spines["top"].set_visible(False)
    ax_histx_0.set(
        xlabel="Propensity score",
    )
    ax.set_xticks(bins, minor=False)
    ax.set_xlim((0, 1))
    observed_only_label = " observed " if observed_outcomes_only else " "
    y_label = f"Mean{observed_only_label}outcome"
    if normalize_y:
        y_label += "\n(standardized)"
    ax.set(xlabel="", ylabel=y_label)
    ax.grid(axis="x", linewidth=5)
    plt.setp(ax.get_xticklabels(), visible=False)
    # add nTV
    n_tv = normalized_total_variation(
        propensity_scores=ps, treatment_probability=a.mean()
    )
    ax.text(
        x=0.9, y=0.9, transform=plt.gcf().transFigure, s=f"nTV: {n_tv:.4f}", fontsize=20
    )
    return ax, [ax_histx_0, ax_histx_1]


def get_mean_outcomes_by_ps_bin(
    y_0: np.array,
    y_1: np.array,
    ps: np.array,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[np.array, np.array, np.array]:

    """
    Compute mean outcome values for each intervention by propensity bin

    Parameters
    ----------
    y_0 : outcomes for control units
    y_1 : outcomes for treated units
    ps : propensity scores
    n_bin : int, optional
        number of bins, by default 10

    Returns
    -------
    Tuple[np.array, np.array, np.array, np.array]
        - Mean outcome values per bin for each outcome [n_bins, 2]
        - Standard error of mean outcome values per bin for each outcome [n_bins, 2]
        - Counts value per each bin for each outcome
        - bins mean ps values
    """

    check_consistent_length(ps, y_0, y_1)

    bins_df = pd.DataFrame({"ps": ps, "y_0": y_0, "y_1": y_1})
    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(ps, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    bins_df["ps_bin"] = pd.cut(x=bins_df["ps"], bins=bins, labels=bins[1:])
    ps_0 = (
        bins_df.loc[~bins_df["y_0"].isna()]
        .groupby("ps_bin", sort=True)["ps"]
        .agg(np.mean)
    )
    ps_1 = (
        bins_df.loc[~bins_df["y_1"].isna()]
        .groupby("ps_bin", sort=True)["ps"]
        .agg(np.mean)
    )
    bins_mean_ps = pd.concat([ps_0, ps_1], axis=1).sort_index()

    bins_mean_outcomes = bins_df.groupby("ps_bin", sort=True).agg(
        **{
            "mean_y_0": pd.NamedAgg("y_0", np.nanmean),
            "mean_y_1": pd.NamedAgg("y_1", np.nanmean),
        }
    )

    def _sem(x):
        "scipy.stats.sem does not deal correctly with NaNs, return masked"
        return np.nanstd(x, ddof=1) / np.sqrt((1 - np.isnan(x)).sum())

    bins_sem_outcomes = bins_df.groupby("ps_bin", sort=True).agg(
        **{
            "sem_y_0": pd.NamedAgg("y_0", _sem),
            "sem_y_1": pd.NamedAgg("y_1", _sem),
        }
    )
    bins_sem_outcomes.fillna(0, inplace=True)

    def _nancount(x):
        return (1 - np.isnan(x)).sum()

    bins_counts = bins_df.groupby("ps_bin", sort=True).agg(
        **{
            "count_y_0": pd.NamedAgg("y_0", _nancount),
            "count_y_1": pd.NamedAgg("y_1", _nancount),
        }
    )
    for b in bins[1:]:
        if b not in bins_mean_outcomes.index:
            bins_mean_outcomes.loc[b] = [np.nan, np.nan]
        if b not in bins_mean_ps.index:
            bins_mean_ps.loc[b] = [np.nan, np.nan]
        if b not in bins_counts.index:
            bins_counts.loc[b] = [0, 0]

    assert (
        bins_mean_ps.shape[0] == n_bins
    ), f"bins_mean should be of shape [n_bins={n_bins}, 2],  got {bins_mean_ps.shape[0]} at first dimension"
    return (
        bins_mean_outcomes.values,
        bins_sem_outcomes.values,
        bins_counts.values,
        bins_mean_ps.values,
    )

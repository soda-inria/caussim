# ### Plot and utils functions for 2D simulations ### #
###   Plot utils # ###
from typing import Dict, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline

from caussim.config import (
    COLOR_MAPPING,
    DIR2FIGURES,
    LABEL_MAPPING,
    LS_MAPPING,
    TAB_COLORS,
)
from caussim.data.causal_df import CausalDf
from caussim.data.simulations import CausalSimulator, get_transformed_space_from_simu
from caussim.estimation.estimation import get_selection_metrics
from caussim.estimation.estimators import (
    SLEARNER_LABEL,
    AteEstimator,
    get_meta_learner_components,
)
from caussim.experiences.pipelines import CLASSIFIERS, REGRESSORS
from caussim.pdistances.mmd import mmd_rbf, normalized_total_variation
from caussim.reports.plots_utils import CAUSAL_METRIC_LABELS
from caussim.utils import dic2str


# TODO: completely deprecated but used in toy example...
def estimate_DML(train_df: CausalDf, estimation_config, test_df: CausalDf = None):
    """
    Args:
        train_df ([type]): [description]
        estimation_config ([type]): Dictionnary with at least model keys :
            - outcome_model,
            - propensity_model,
        test_df ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    outcome_model = clone(estimation_config.get("outcome_model"))
    if not hasattr(outcome_model, "named_steps"):
        outcome_model = make_pipeline(outcome_model)
    propensity_model = clone(estimation_config.get("propensity_model"))
    if not hasattr(propensity_model, "named_steps"):
        propensity_model = make_pipeline(propensity_model)
    if estimation_config.get("calibration", False):
        propensity_model = CalibratedClassifierCV(base_estimator=propensity_model, cv=3)
    X_cols = [col for col in train_df.columns if col.lower().startswith("x")]
    estimator = AteEstimator(
        outcome_model=outcome_model,
        propensity_model=propensity_model,
        n_splits=estimation_config.get("n_splits", 5),
        clip=estimation_config.get("clip", 1e-3),
        random_state_cv=estimation_config.get("random_state_cv", 0),
        meta_learner=estimation_config.get("meta_learner", SLEARNER_LABEL),
    )
    X_full = np.concatenate([train_df["a"].values, train_df[X_cols].values])
    train_predictions, train_metrics = estimator.fit_predict(
        X=X_full, y=train_df["y"].values
    )
    if test_df is None:
        test_df = train_df
    test_predictions, test_metrics = estimator.predict(
        X=X_full,
        y=test_df["y"].values,
        leftout=True,
    )

    # add oracle to predictions results
    test_predictions["oracle_mu_0"] = test_df["mu_0"]
    test_predictions["oracle_mu_1"] = test_df["mu_1"]
    test_predictions["oracle_e"] = test_df["e"]
    # estimate for different kind of tau
    estimates = {}
    for tau_name in ["REG", "IPW", "AIPW"]:
        estimator.tau = tau_name
        tau_estimation = estimator.estimate(test_predictions)
        for k, v in tau_estimation.items():
            estimates[f"{tau_name}_{k}"] = v

    true_estimates = test_df.estimate_oracles()
    all_estimates = {**true_estimates, **estimates}
    # estimation of tau_risk
    if outcome_model is not None:
        all_estimates["tau_risk"] = mean_squared_error(
            all_estimates["cate"], all_estimates["REG_hat_cate"], squared=True
        )
    return test_predictions, all_estimates, test_metrics, estimator


def show_full_sample(
    df: pd.DataFrame,
    fig=None,
    cols=["x_0", "y"],
    show_sample=True,
    show_mu_oracle=True,
    show_e_oracle=False,
    show_hat_e=False,
    show_hat_y=False,
    show_DM=False,
    show_tau=False,
    axes_labels=[
        "Confounding X",
        "Outcome Y",
    ],  # [r"$X = Charlson \; score$", r"$Y = \mathbb{P}[Mortality]$"],
    legend=False,
    legend_prop=None,
    ylims=None,
    xlims=None,
    common_norm=True,
    pretty_axes_locator=False,
    show_ntv=True,
) -> Tuple[plt.Axes, plt.Axes]:

    df["w_a"] = df["a"] * 1 / df["e"] + (1 - df["a"]) / (1 - df["e"])
    max_size = np.max(df["w_a"]) - np.min(df["w_a"])
    marker_scale = 1000 / max_size
    # define grid depending on showing sample distribution
    if fig is None:
        fig = plt.figure()
    if show_sample:
        gs = fig.add_gridspec(
            2, 1, height_ratios=(7, 2), bottom=0.1, top=0.9, hspace=0.05
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_histx = fig.add_subplot(gs[1, 0], sharex=ax)
    else:
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax_histx = None
    for a, m in zip((0, 1), ("o", "o")):
        mask_a = df["a"] == a
        part = df[mask_a]
        if show_e_oracle:
            ss = df.loc[mask_a, "w_a"].values
            ss *= marker_scale
        else:
            ss = 200
        if show_sample:
            ax.scatter(
                part[cols[0]].values,
                part[cols[1]].values,
                marker=m,
                edgecolors=None,
                linewidth=0,
                s=ss,
                facecolors=COLOR_MAPPING[a],
                alpha=0.7,
            )
        if show_mu_oracle:
            mu_a = df[["x_0", f"mu_{a}"]].sort_values("x_0")
            ax.plot(
                mu_a["x_0"].values,
                mu_a[f"mu_{a}"].values,
                linewidth=4,
                linestyle=LS_MAPPING[a],
                c=COLOR_MAPPING[a],
            )
        # plot estimates
        if show_hat_e:
            df["hat_w_a"] = df["a"] * 1 / df["hat_e"] + (1 - df["a"]) / (
                1 - df["hat_e"]
            )
            ss_hat_e = df.loc[mask_a, "hat_w_a"].values * marker_scale
            ax.scatter(
                part[cols[0]].values,
                part[cols[1]].values,
                marker=m,
                s=ss_hat_e,
                edgecolors="black",
                facecolors="none",
                linewidth=0.5,
                alpha=1,
            )
        if show_hat_y:
            hat_mu_a = df[["x_0", f"hat_mu_{a}"]].sort_values("x_0")
            ax.plot(
                hat_mu_a["x_0"].values,
                hat_mu_a[f"hat_mu_{a}"].values,
                linewidth=3,
                linestyle="-",
                c="black",
            )
    if show_tau:
        x_tau = df.sort_values("x_0")["x_0"].values[int(0.75 * df.shape[0])]
        mu_1_at_x_tau = df.loc[df["x_0"] == x_tau, "mu_1"].values[0]
        mu_0_at_x_tau = df.loc[df["x_0"] == x_tau, "mu_0"].values[0]
        arrow = mpatches.FancyArrowPatch(
            (x_tau, mu_0_at_x_tau),
            (x_tau, mu_1_at_x_tau),
            mutation_scale=30,
            color="black",
        )
        ax.add_patch(arrow)
        mu_at_x_tau_min = np.min([mu_1_at_x_tau, mu_0_at_x_tau])
        mu_at_x_tau_max = np.max([mu_1_at_x_tau, mu_0_at_x_tau])
        ax.text(
            x_tau + x_tau / 20,
            mu_at_x_tau_min + (mu_at_x_tau_max - mu_at_x_tau_min) / 2,
            s=r"$\tau(x)$",
            fontsize=int(ss / 5),
        )
    # add an axes for x distribution
    if show_sample:
        sns.kdeplot(
            ax=ax_histx,
            data=df,
            x=cols[0],
            hue="a",
            palette=COLOR_MAPPING,
            common_norm=common_norm,
            alpha=0.6,
            fill=True,
            # clip=(0, 20),
            legend=False,
        )
        ax_histx.tick_params(axis="y", labelleft=False)
        ax_histx.set_ylabel("")
        ax_histx.grid(False)
        ax_histx.set_xlabel(axes_labels[0])
        plt.setp(ax.get_xticklabels(), visible=False)
    else:
        ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    if legend:
        # outcome legend
        legend_elements = [
            Line2D(
                [0],
                [0],
                color=COLOR_MAPPING[0],
                linestyle=":",
                lw=4,
            ),
            Line2D(
                [0],
                [0],
                color=COLOR_MAPPING[1],
                linestyle=":",
                lw=4,
            ),
        ]
        legend_y_labels = [r"Untreated outcome $Y_0 (x)$", r"Treated outcome $Y_1 (x)$"]
        if show_hat_y:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    lw=3,
                    linestyle="-",
                )
            )
            legend_y_labels.append(r"$\hat \mu_a (x)$")
        if not show_e_oracle:
            legend_treatments = ax.legend(
                handles=legend_elements,
                labels=legend_y_labels,
                loc="upper left",
                numpoints=1,
                handler_map={tuple: HandlerTuple(ndivide=2)},
                prop=legend_prop,
            )
            ax.add_artist(legend_treatments)
        # ps legend
        legend_ps_handles = []
        legend_ps_labels = []
        legend_marker_size = 10
        if show_e_oracle:
            legend_ps_handles.append(
                (
                    Line2D(
                        [0.1],
                        [0.5],
                        marker="o",
                        color=COLOR_MAPPING[0],
                        alpha=0.6,
                        linewidth=0,
                        markersize=legend_marker_size,
                    ),
                    Line2D(
                        [0.3],
                        [0.2],
                        marker="o",
                        color=COLOR_MAPPING[1],
                        alpha=0.6,
                        linewidth=0,
                        markersize=legend_marker_size,
                    ),
                ),
            )
            legend_ps_labels.append(r"$w(x, a)=\frac{a}{e(x)} + \frac{1-a}{1-e(x)}$")
            if show_hat_e:
                legend_ps_labels.append(r"$\hat w(x, a)$")
                legend_ps_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        markerfacecolor="none",
                        markeredgecolor="black",
                        linewidth=0.5,
                        markersize=legend_marker_size,
                        ls="",
                    )
                )
            legend_ps = ax.legend(
                handles=legend_ps_handles,
                labels=legend_ps_labels,
                loc="lower left",
                title=r"Inverse Propensity weight",
                numpoints=1,
                handler_map={tuple: HandlerTuple(ndivide=2)},
                prop={"size": 20},
            )
            ax.add_artist(legend_ps)
    if show_DM:
        mean_1 = np.mean(df[df["a"] == 1]["y"].values)
        mean_0 = np.mean(df[df["a"] == 0]["y"].values)
        ax.axhline(mean_1, c=COLOR_MAPPING[1], lw=3)
        ax.text(5.5, mean_1, s=r"$E [Y|A=1]$", c="black", ha="left")
        ax.axhline(mean_0, c=COLOR_MAPPING[0], lw=3)
        ax.text(5.5, mean_0, s=r"$E [Y|A=0]$", c="black", ha="left")
    if ylims is None:
        ymin, ymax = np.min(df["y"]), np.max(df["y"])
    else:
        ymin, ymax = ylims
    if xlims is None:
        xmin_lim, xmax_lim = np.floor(np.min(df["x_0"])), np.ceil(np.max(df["x_0"]))
    else:
        xmin_lim, xmax_lim = xlims
    ax.set_ylim((ymin, ymax))
    ax.set_xlim((xmin_lim, xmax_lim))
    if pretty_axes_locator:
        y_formatter = FixedFormatter(["0", "", "", "", "", "1"])
        y_locator = FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.yaxis.set_major_formatter(y_formatter)
        ax.yaxis.set_major_locator(y_locator)
        xmin = np.floor(np.min(df["x_0"]))
        xmax = np.max(df["x_0"])
        xmed = (xmax - xmin) / 2
        x_formatter = FixedFormatter(
            [f"{xmin:.0f}", "", f"{xmed:.0f}", "", f"{xmax:.0f}"]
        )
        ax.xaxis.set_major_formatter(x_formatter)
        x_locator = FixedLocator(
            [xmin, (xmed - xmin) / 2, xmed, (xmax - xmed) / 2 + xmed, xmax]
        )
        ax.xaxis.set_major_locator(x_locator)
    if show_sample:
        xmin = np.floor(np.min(df["x_0"]))
        ax.set(xlabel=None)
        covariates_distrib_label = "Populations covariate distributions"
        ax_histx.text(xmin + 0.05, 0.005, covariates_distrib_label, fontweight="heavy")
    if show_ntv:
        ntv = normalized_total_variation(df["e"], df["a"].mean())
        ax.text(0, 1, s=f"NTV: {ntv:.3f}", transform=plt.gcf().transFigure, fontsize=20)
    return ax, ax_histx


def show_outcome_fit(
    df: pd.DataFrame,
    ax,
    outcome_model,
    propensity_model,
    show_sample=True,
    show_e_oracle=False,
    show_mu_oracle=True,
    show_hat_y=False,
    show_hat_e=False,
    n_splits=5,
    clip=1e-2,
    calibration=True,
    random_state_estimation=0,
    legend_prop=None,
):
    sample_ = df.copy()
    if type(outcome_model) == str:
        outcome_model = REGRESSORS[outcome_model]
    else:
        outcome_model = outcome_model
    if type(propensity_model) == str:
        propensity_model = CLASSIFIERS[propensity_model]
    else:
        propensity_model = propensity_model
    estimation_config = {
        "outcome_model": outcome_model,
        "propensity_model": propensity_model,
        "n_splits": n_splits,
        "clip": clip,
        "calibration": calibration,
        "random_state_cv": random_state_estimation,
    }
    test_predictions, all_estimates, test_metrics, estimator = estimate_DML(
        train_df=sample_, estimation_config=estimation_config
    )
    ## plotting stuff
    sample_ = sample_.merge(
        pd.DataFrame(
            {
                "hat_e": estimator.predictions["hat_a_proba"],
            },
            index=estimator.predictions["idx"],
        ),
        left_index=True,
        right_index=True,
    )

    dummy_ya = np.random.randint(0, 2, size=len(sample_["x_0"]))
    predictions, _ = estimator.predict(
        sample_["x_0"].values.reshape(-1, 1), dummy_ya, leftout=True
    )
    sample_["hat_mu_0"] = predictions["hat_mu_0"]
    sample_["hat_mu_1"] = predictions["hat_mu_1"]
    sample_["hat_e"] = np.clip(sample_["hat_e"], clip, 1 - clip)

    ax, ax_histx = show_full_sample(
        sample_,
        ax,
        show_sample=show_sample,
        show_e_oracle=show_e_oracle,
        show_mu_oracle=show_mu_oracle,
        show_hat_e=show_hat_e,
        show_hat_y=show_hat_y,
        legend=True,
        legend_prop=legend_prop,
    )

    return estimator, all_estimates, test_metrics, ax, ax_histx


def show_estimates(
    ax,
    estimations,
    title="",
    metrics=None,
    tau=True,
    tau_DM=False,
    abs_bias_ATE=False,
    tau_IPW_oracle=False,
    tau_G=False,
    tau_IPW=False,
    tau_AIPW=False,
    tau_risk=False,
    oracle_r_risk=False,
    oracle_r_risk_rewritten=False,
    oracle_mu_iptw_risk=False,
    oracle_r_risk_ipw=False,
    oracle_r_risk_IS2=False,
    r_risk_IS2=False,
    mu_risk=False,
    outcome_r2=False,
    propensity_auroc=False,
    propensity_bss=False,
    x_anchor=1.02,
    y_anchor=0.01,
    fontsize=20,
    color="black",
):
    """Add metrics annotation to a plot"""
    # TODO: add float precision as argument
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    if tau:
        estimation_txt = r"""Estimates (\%)
${\tau}=$ """ + "{:.2f}".format(
            100 * estimations["ate"],
        )
    else:
        estimation_txt = ""
    if tau_DM:
        estimation_txt += r"""
$\widehat{{\tau}}_{{DM}}=$ """ + "{:.2f}".format(
            100 * estimations["hat_ate_diff"]
        )
    if tau_IPW_oracle:
        estimation_txt += r"""
$\widehat{\tau}^*_{IPW}(e)=$ """ + "{:.2f}".format(
            100 * estimations["oracle_ipw"]
        )
    if tau_G:
        estimation_txt += r"""
$\widehat{\tau}_{\hat f}=$ """ + "{:.2f}".format(
            100 * estimations["REG_hat_ate"]
        )

    if tau_IPW:
        estimation_txt += r"""
$\widehat{\tau}_{IPW}(\hat e)=$ """ + "{:.2f}".format(
            100 * estimations["IPW_hat_ate"]
        )
    if tau_AIPW:
        estimation_txt += r"""
$\widehat{\tau}_{AIPW}(\hat f, \hat e)=$ """ + "{:.2f}".format(
            100 * estimations["AIPW_hat_ate"]
        )
    if (
        outcome_r2
        | propensity_auroc
        | propensity_bss
        | tau_risk
        | abs_bias_ATE
        | oracle_r_risk
        | oracle_r_risk_rewritten
        | oracle_mu_iptw_risk
        | oracle_r_risk_ipw
        | r_risk_IS2
        | oracle_r_risk_IS2
        | mu_risk
    ):
        if tau:
            estimation_txt += "\n\n"
        estimation_txt += "Metrics:"
        if abs_bias_ATE:
            estimation_txt += r"""
$| \tau - \widehat{\tau}_{\hat f)} |=$ """ + r"{:.2f} \%".format(
                100 * np.abs(estimations["ate"] - estimations["REG_hat_ate"])
            )

        if tau_risk:
            estimation_txt += r"""
$\tau\mathrm{-risk}(\hat f)=$ """ + "{:.2f}".format(
                estimations["tau_risk"]
            )
        if oracle_r_risk:
            estimation_txt += r"""
$R\mathrm{-risk}^*(\hat f)=$ """ + "{:.2f}".format(
                estimations["oracle_r_risk"]
            )
        if oracle_r_risk_rewritten:
            estimation_txt += r"""
$R\mathrm{-risk}_r^*(\hat f)=$ """ + "{:.2f}".format(
                estimations["oracle_r_risk_rewritten"]
            )
        if oracle_r_risk_ipw:
            estimation_txt += r"""
$R \mathrm{-risk}^*_{IPW}=$ """ + "{:.2f}".format(
                estimations["oracle_r_risk_ipw"]
            )
        if oracle_r_risk_IS2:
            estimation_txt += r"""
$R \mathrm{-risk}^*_{IS2}=$ """ + "{:.2f}".format(
                estimations["oracle_r_risk_IS2"]
            )
        if oracle_mu_iptw_risk:
            estimation_txt += r"""
$\mu \mathrm{-risk}^*_{IPW}=$ """ + "{:.2f}".format(
                estimations["oracle_mu_iptw_risk"]
            )
        if r_risk_IS2:
            estimation_txt += r"""
$R \mathrm{-risk}_{IS2}=$ """ + "{:.2f}".format(
                estimations["r_risk_IS2"]
            )
        if mu_risk:
            estimation_txt += r"""
$\mu \mathrm{-risk}(\hat f)=$ """ + "{:.2f}".format(
                estimations["mu_risk"]
            )
        if outcome_r2:
            estimation_txt += r"""
$\widehat{R2}(\hat f)=$ """ + "{:.2f}".format(
                metrics["outcome_r2"]
            )
        if propensity_auroc:
            estimation_txt += r"""
OC(\hat e, a)=$ """ + "{:.2f}".format(
                metrics["propensity_auroc"]
            )
        if propensity_bss:
            estimation_txt += r"""
SS(\hat e, a)=$ """ + "{:.2f}".format(
                metrics["propensity_bss"]
            )
    # place a text box in upper left in axes coords
    estimation_txt = title + "\n" + estimation_txt
    ax.text(
        x_anchor,
        y_anchor,
        estimation_txt,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        bbox=props,
        color=color,
    )

    return ax


def plot_simu2D(
    df: pd.DataFrame, sim: CausalSimulator, fig=None, ax=None, legend=True, title=True
):
    """
    Plot a 2D plot of the simulated data.

    Args:
        df (pd.DataFrame): [description]
        sim (CausalSimulator): [description]
        fig ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    nx, ny = (14, 14)
    xx_grid, yy_grid = np.meshgrid(
        np.linspace(df["x_0"].min(), df["x_0"].max(), nx),
        np.linspace(df["x_1"].min(), df["x_1"].max(), ny),
    )
    X_grid = np.c_[xx_grid.ravel(), yy_grid.ravel()]

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        save = True
    else:
        save = False
    for a, s, cmap in zip([0, 1], ["o", "o"], ["Oranges", "Blues"]):
        ax.plot(
            df.query(f"a =={a}")["x_0"],
            df.query(f"a =={a}")["x_1"],
            s,
            c=COLOR_MAPPING[a],
            label=LABEL_MAPPING[a],
        )
        ax.contour(
            xx_grid,
            yy_grid,
            sim.mu(X_grid, np.full(X_grid.shape[0], a)).reshape((nx, ny)),
            cmap=cmap,
            levels=20,
        )
    ## mmd calculation (work only if baseline_pipeline and tau pipeline shares same representation space)
    ### TODO: general case where both are different and we should concatenate

    x_cols = [col for col in df.columns if col.lower().startswith("x_")]
    mmd_original_space = mmd_rbf(df.query("a == 0")[x_cols], df.query("a == 1")[x_cols])

    Z_control, Z_treated = get_transformed_space_from_simu(sim, df)
    mmd_feature_space_populations = mmd_rbf(Z_control, Z_treated)
    if legend:
        ax.legend(
            title=r"Treatment Status",
            borderaxespad=0,
            ncol=1,
            loc="upper right",
            bbox_to_anchor=(0.9, 1.01),
            transform=plt.gcf().transFigure,
            fontsize=20,
        )
    if title:
        ax.set_title(
            f"2D plot of response surfaces, mu_0={sim.baseline_link_type}, tau={sim.effect_link_type}, \n overlap={sim.overlap}, effect size={sim.effect_size}, \n mmd original space={mmd_original_space:.2E}, mmd feature space={mmd_feature_space_populations:.2E}"
        )
    # ax.set_aspect("equal")
    if save:
        fig.tight_layout()
        plt.savefig(
            DIR2FIGURES
            / f"plot2D_a={sim.treatment_assignment_type}__mu0={dic2str(sim.baseline_pipeline.get_params())}__tau={dic2str(sim.effect_pipeline.get_params())}__overlap={sim.overlap}__effect_size={sim.effect_size}_rs_gaussian={sim.rs_gaussian}.png",
            bbox_inches="tight",
        )
    return fig, ax


def plot_simu1D_cut(
    df: pd.DataFrame,
    sim: CausalSimulator,
    cut_points: np.array = None,
    fig=None,
    ax=None,
):
    """
    Get 1D view of causal simulator response surfaces by cutting along a specific axis
    Args:
        df (pd.DataFrame): [description]
        sim (CausalSimulator): [description]
        cut_points (np.array, optional): [description]. Defaults to None.
        fig ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    x_grid, lambdas = get_cut(df, sim, cut_points=cut_points)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for a, s in zip([0, 1], ["o", "+"]):
        ax.plot(
            lambdas,
            sim.mu(x_grid, np.full(x_grid.shape[0], a)),
            c=COLOR_MAPPING[a],
            lw=4,
        )
    # ax.set_title("Cut direction: {}".format(cut_points))
    return fig, ax


def get_cut(df, sim, cut_points: np.array = None):
    if cut_points is None:
        cut_points = (
            sim.baseline_pipeline.pipeline.named_steps.featurization.components_
        )
        base_ix = np.random.choice(np.arange(cut_points.shape[0]), 2, replace=False)
        cut_points = cut_points[base_ix]
    cut_direction = cut_points[0, :] - cut_points[1, :]
    xmax, xmin = df["x_0"].max(), df["x_0"].min()
    ymax, ymin = df["x_1"].max(), df["x_1"].min()
    xmax_lambda = (xmax - cut_points[0, 0]) / (cut_direction[0] + 1e-12)
    xmin_lambda = (xmin - cut_points[0, 0]) / (cut_direction[0] + 1e-12)
    ymax_lambda = (ymax - cut_points[0, 1]) / (cut_direction[1] + 1e-12)
    ymin_lambda = (ymin - cut_points[0, 1]) / (cut_direction[1] + 1e-12)
    lambda_max = max(ymax_lambda, xmax_lambda)
    lambda_min = min(ymin_lambda, xmin_lambda)
    lambda_grid = np.linspace(lambda_min, lambda_max, 100)
    x_grid = [t * cut_direction for t in lambda_grid] + cut_points[0, :]
    return x_grid, lambda_grid


def plot_simu_2D_1D_cuts(df, sim, fig=None):
    """Plot 2D and 1D cuts of the response surface for a given simulation and sample data

    Args:
        sim ([type]): [description]
        df ([type]): [description]
        fig ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if fig is None:
        fig = plt.figure(figsize=(12, 12))
    basis = sim.baseline_pipeline.pipeline.named_steps.featurization.components_
    cut_basis_line, _ = get_cut(df, sim, basis)
    barycentre = basis.mean(axis=0)
    orthogonal_dir = np.array([basis[0, 1] - basis[1, 1], -basis[0, 0] + basis[1, 0]])
    basis_orthog = np.vstack([barycentre, barycentre + orthogonal_dir])
    cut_ortho_line, _ = get_cut(df, sim, basis_orthog)

    ax1 = plt.subplot(211)

    fig, ax1 = plot_simu2D(df, sim, fig=fig, ax=ax1)
    ax1.plot(cut_basis_line[:, 0], cut_basis_line[:, 1], c="black", lw=2)
    ax1.plot(cut_ortho_line[:, 0], cut_ortho_line[:, 1], c="black", lw=2)
    ax1.set_xlim(df["x_0"].min(), df["x_0"].max())
    ax1.set_ylim(df["x_1"].min(), df["x_1"].max())
    ax1.set_title("2D view of response surfaces")
    # ax1.set_aspect("equal")

    ax2 = plt.subplot(223)
    ax2.set_title("1D cut")
    fig, _ = plot_simu1D_cut(df, sim, cut_points=basis, fig=fig, ax=ax2)

    ax3 = plt.subplot(224)
    ax3.set_title("1D cut orthogonal")
    fig, _ = plot_simu1D_cut(df, sim, cut_points=basis_orthog, fig=fig, ax=ax3)
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, [ax1, ax2, ax3]


# ### Plot risk maps ### #
def plot_mu_risk(
    df,
    sim,
    estimator,
    predictions,
    metrics,
    risk="mu",
    plot_components=True,
    max_quantile=0.9,
    fig=None,
    ax=None,
):
    """
    Plot mu risks for both population in a 2D causal simulation.

    mu_risk can be reweighted with propensity score (oracle or with prefitted nuisance model)

    Args:
        df ([type]): [description]
        sim ([type]): [description]
        estimator ([type]): [description]
        predictions ([type]): [description]
        metrics ([type]): [description]
        risk (str, optional): [description]. Defaults to "mu".
        plot_components (bool, optional): [description]. Defaults to True.
        max_quantile (float, optional): [description]. Defaults to 0.9.
        fig ([type], optional): [description]. Defaults to None.
        axes ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    available_risks = ["mu", "mu_iptw", "mu_iptw_oracle"]

    mu_0_residuals_data = (predictions["hat_mu_0"] - df["mu_0"]) ** 2
    mu_1_residuals_data = (predictions["hat_mu_1"] - df["mu_1"]) ** 2

    nx, ny = (50, 50)
    xx_grid, yy_grid = np.meshgrid(
        np.linspace(df["x_0"].min(), df["x_0"].max(), nx),
        np.linspace(df["x_1"].min(), df["x_1"].max(), ny),
    )
    X_grid = np.c_[xx_grid.ravel(), yy_grid.ravel()]

    # z variables
    mu_0 = sim.response_surface(X_grid, np.zeros(X_grid.shape[0]))
    mu_1 = sim.response_surface(X_grid, np.ones(X_grid.shape[0]))
    grid_predictions = estimator.predict(X=X_grid, A=None)
    mu_0_residuals = (mu_0 - grid_predictions["hat_mu_0"]) ** 2
    mu_1_residuals = (mu_1 - grid_predictions["hat_mu_1"]) ** 2
    if risk == "mu":
        sup_labels = [
            rf"($\mu_{{}}$ - $\hat \mu_{{}})^2$".format(mu_ix, mu_ix)
            for mu_ix in [0, 1]
        ]
    elif risk == "mu_iptw_oracle":
        e_oracle = sim.propensity_density(X_grid)
        mu_0_residuals = mu_0_residuals / (1 - e_oracle)
        mu_1_residuals = mu_1_residuals / e_oracle
        sup_labels = [
            rf"($\mu_{{}}$ - $\hat \mu_{{}})^2 / p(A={mu_ix}|X=x)$".format(
                mu_ix, mu_ix, mu_ix
            )
            for mu_ix in [0, 1]
        ]
        mu_0_residuals_data = mu_0_residuals_data / (1 - df["e"])
        mu_1_residuals_data = mu_1_residuals_data / df["e"]
    elif risk == "mu_iptw":
        mu_0_residuals = mu_0_residuals / (1 - grid_predictions["check_e"])
        mu_1_residuals = mu_1_residuals / grid_predictions["check_e"]
        sup_labels = [
            rf"($\mu_{{}}$ - $\hat \mu_{{}})^2 / \hat p(A={mu_ix}|X=x)$".format(
                mu_ix, mu_ix, mu_ix
            )
            for mu_ix in [0, 1]
        ]
        mu_0_residuals_data = mu_0_residuals_data / (1 - predictions["check_e"])
        mu_1_residuals_data = mu_1_residuals_data / predictions["check_e"]
    elif risk == "check_e":
        mu_0_residuals = 1 / (1 - grid_predictions["check_e"])
        mu_1_residuals = 1 / grid_predictions["check_e"]
        sup_labels = [
            rf" \hat p(A={mu_ix}|X=x)$".format(mu_ix, mu_ix, mu_ix) for mu_ix in [0, 1]
        ]
        mu_0_residuals_data = 1 / (1 - predictions["check_e"])
        mu_1_residuals_data = 1 / predictions["check_e"]
    else:
        raise ValueError(
            f"Got {risk} but support only following risks:\n {available_risks}"
        )
    mu_0_residuals = mu_0_residuals / mu_0_residuals.sum()
    mu_1_residuals = mu_1_residuals / mu_1_residuals.sum()
    zmax_0 = (mu_0_residuals_data / mu_0_residuals_data.sum()).quantile(max_quantile)
    zmax_1 = (mu_1_residuals_data / mu_1_residuals_data.sum()).quantile(max_quantile)
    if risk in ["mu_iptw_oracle", "mu_iptw"]:
        mu_0_residuals[mu_0_residuals > zmax_0] = zmax_0
        mu_1_residuals[mu_1_residuals > zmax_1] = zmax_1
    # zmaxs = [zmax_0, zmax_1]*
    zmaxs = [mu_0_residuals.max(), mu_1_residuals.max()]
    # cap grid residuals to data residuals
    mu_residuals = [mu_0_residuals, mu_1_residuals]

    if fig is None:
        fig, ax = plt.subplots(1, 2, figsize=(20, 12))
        save = True
    else:
        save = False
    cmaps = ["Oranges", "Blues"]
    for mu_ix, ax in enumerate(ax):
        for col, a, s in zip([COLOR_MAPPING[0], COLOR_MAPPING[1]], [0, 1], ["o", "s"]):
            ax.plot(
                df.query(f"a =={a}")["x_0"],
                df.query(f"a =={a}")["x_1"],
                s,
                markersize=5,
                markeredgewidth=5,
                c=col,
                label=LABEL_MAPPING[a],
                alpha=0.5,
            )
        if plot_components:
            components = get_meta_learner_components(estimator.meta_learner).values()[
                mu_ix
            ]
            ax.plot(
                components[:, 0],
                components[:, 1],
                "x",
                c="green",
                markersize=15,
                markeredgewidth=5,
            )
        z = mu_residuals[mu_ix].values.reshape((nx, ny))
        plt.grid(False)
        c = ax.pcolormesh(
            xx_grid, yy_grid, z, cmap="viridis", vmin=0, vmax=zmaxs[mu_ix]
        )
        plt.colorbar(c, ax=ax)
        ax.set_title(sup_labels[mu_ix])

    model = type(estimator.outcome_model.named_steps["regression"]).__name__
    meta_learner = estimator.meta_learner_name
    tau_risk = metrics["tau_risk"].round(4)
    fig.suptitle(
        f"Residuals for each response surface \n meta_leaner={meta_learner}, _model={model}_R_tau={tau_risk} \n baseline_link={sim.baseline_link_type}, overlap={sim.overlap}",
        y=-0.01,
    )
    if save:
        fig.tight_layout()
        plt.savefig(
            DIR2FIGURES
            / f"mu_residuals_meta={meta_learner}_model={model}_a={sim.treatment_assignment_type}_train_seed={estimator.outcome_model.get_params()['featurization__random_state']}_overlap={sim.overlap}_effect_size={sim.effect_size}_seed={sim.random_seed}.png",
            bbox_inches="tight",
        )
    return fig, ax


def plot_tau_risk(df, sim, estimator, predictions, max_quantile=1, fig=None, ax=None):
    """
    Plot tau risk for a 2D causal simulation.

    Args:
        df ([type]): [description]
        sim ([type]): [description]
        estimator ([type]): [description]
        predictions ([type]): [description]
        fig ([type], optional): [description]. Defaults to None.
        axes ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    risk_data = (df["mu_1"] - df["mu_0"] - predictions["hat_tau"]) ** 2
    nx, ny = (50, 50)
    xx_grid, yy_grid = np.meshgrid(
        np.linspace(df["x_0"].min(), df["x_0"].max(), nx),
        np.linspace(df["x_1"].min(), df["x_1"].max(), ny),
    )
    X_grid = np.c_[xx_grid.ravel(), yy_grid.ravel()]

    # z variable
    mu_0 = sim.response_surface(X_grid, np.zeros(X_grid.shape[0]))
    mu_1 = sim.response_surface(X_grid, np.ones(X_grid.shape[0]))
    grid_predictions = estimator.predict(X=X_grid, A=None)
    tau_risk_x = (
        (mu_1 - mu_0) - (grid_predictions["hat_mu_1"] - grid_predictions["hat_mu_0"])
    ) ** 2
    tau_risk_x = tau_risk_x / tau_risk_x.sum()
    zmax = (risk_data / risk_data.sum()).quantile(max_quantile)
    zmax = tau_risk_x.max()
    # upper bound the tau risk values for the grid
    # tau_risk_x[tau_risk_x > zmax] = zmax
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        save = True
    else:
        save = False
    cmaps = ["Oranges", "Blues"]
    for col, a, s in zip([COLOR_MAPPING[0], COLOR_MAPPING[1]], [0, 1], ["o", "s"]):
        ax.plot(
            df.query(f"a =={a}")["x_0"],
            df.query(f"a =={a}")["x_1"],
            s,
            markersize=5,
            markeredgewidth=5,
            c=col,
            label=LABEL_MAPPING[a],
            alpha=0.5,
        )
    z = tau_risk_x.values.reshape((nx, ny))
    plt.grid(False)
    c = ax.pcolormesh(xx_grid, yy_grid, z, cmap="viridis", vmin=0, vmax=zmax)
    fig.colorbar(c, ax=ax)
    ax.set_title(r"$(\hat \tau_(x) - \hat \tau_{\hat f})^2$")
    fig.suptitle(
        f"PEHE-risk for each response surface \n meta_leaner={estimator.meta_learner_name}, baseline_link={sim.baseline_link_type}, overlap={sim.overlap}"
    )
    model = type(estimator.outcome_model.named_steps["regression"]).__name__
    if save:
        fig.tight_layout()
        plt.savefig(
            DIR2FIGURES
            / f"tau_risk_mlearner={estimator.meta_learner_name}_model={model}__mu0={dic2str(sim.baseline_pipeline.get_params())}__tau={dic2str(sim.effect_pipeline.get_params())}__o={sim.overlap}__eff={sim.effect_size}_s={sim.random_seed}.png",
            bbox_inches="tight",
        )
    return fig, ax


def plot_models_comparison(
    df: pd.DataFrame,
    predictions_model_ref: pd.DataFrame,
    predictions_model_compared: pd.DataFrame,
    name_model_ref: str = "Model 0",
    name_model_compared: str = "Model 1",
    plot_annotation: bool = True,
    fig=None,
    show_ntv=True,
) -> Tuple[plt.Axes, plt.Axes]:
    """
    Plot a sample of simulated data and compare two causal inference models applied to it.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    prediction_model_ref : pd.DataFrame
        _description_
    prediction_model_compared : pd.DataFrame
        _description_
    name_model_ref : str, optional
        _description_, by default "Model 0"
    name_model_compared : str, optional
        _description_, by default "Model 1"
    fig : _type_, optional
        _description_, by default None

    Returns
    -------
    Tuple[plt.Axes, plt.Axes]
        _description_
    """
    ax, ax_hist = show_full_sample(df=df, show_sample=True, fig=fig, show_ntv=show_ntv)
    models_names = [name_model_ref, name_model_compared]
    models_predictions = [predictions_model_ref, predictions_model_compared]
    model_colors = dict(zip(models_names, ["black", TAB_COLORS[8]]))

    for m_ix, (m_name, m_predictions) in enumerate(
        zip(models_names, models_predictions)
    ):
        for a in [0, 1]:
            hat_mu_a = pd.concat(
                [df["x_0"], m_predictions[f"hat_mu_{a}"]], axis=1
            ).sort_values("x_0")
            ax.plot(
                hat_mu_a["x_0"].values,
                hat_mu_a[f"hat_mu_{a}"].values,
                linewidth=3,
                linestyle=LS_MAPPING[a],
                c=model_colors[m_name],
            )
            if m_name == name_model_ref:
                m_label_ix = 0
                h_alignment = "left"
            else:
                m_label_ix = -1
                h_alignment = "right"
            x_scale = df["x_0"].max() - df["x_0"].min()
            y_scale = df["y"].max() - df["y"].min()
        ax.annotate(
            text=f"Best for {m_name}",
            xy=(
                hat_mu_a["x_0"].values[m_label_ix],
                hat_mu_a[f"hat_mu_1"].values[m_label_ix] + y_scale / 20,
            ),
            fontsize=25,
            color=model_colors[m_name],
            ha=h_alignment,
            va="bottom",
        )
        if plot_annotation:
            metrics = get_selection_metrics(
                causal_df=CausalDf(df), predictions=m_predictions
            )
            show_estimates(
                ax,
                metrics,
                title=m_name,
                tau=False,
                tau_risk=True,
                oracle_r_risk=True,
                oracle_r_risk_IS2=True,
                oracle_mu_iptw_risk=True,
                mu_risk=True,
                y_anchor=-0.2 + m_ix * 0.9,
                color=model_colors[m_name],
            )
    return ax, ax_hist


def plot_metric_comparison(
    df: pd.DataFrame,
    metric_best_predictions: Dict[str, pd.DataFrame],
    metric_ref: str = "tau_risk",
    metric_compared: str = "oracle_r_risk",
    plot_annotation: bool = True,
    fig=None,
    show_ntv=True,
) -> Tuple[plt.Axes, plt.Axes]:
    """
    Wrapper around plot_model_comparison adapated to metric comparison.
    Parameters
    ----------
    df : pd.DataFrame
        _description_
    metric_best_predictions : Dict[str, pd.DataFrame]
        _description_
    metric_ref : str, optional
        _description_, by default "tau_risk"
    metric_compared : str, optional
        _description_, by default "oracle_r_risk"
    plot_annotation : bool, optional
        _description_, by default True
    fig : _type_, optional
        _description_, by default None

    Returns
    -------
    Tuple[plt.Axes, plt.Axes]
        _description_
    """
    ax, ax_hist = plot_models_comparison(
        df=df,
        predictions_model_ref=metric_best_predictions[metric_ref],
        predictions_model_compared=metric_best_predictions[metric_compared],
        name_model_ref=CAUSAL_METRIC_LABELS[metric_ref],
        name_model_compared=CAUSAL_METRIC_LABELS[metric_compared],
        fig=fig,
        plot_annotation=plot_annotation,
        show_ntv=show_ntv,
    )
    return ax, ax_hist

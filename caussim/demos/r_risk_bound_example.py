#%%
# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.utils import check_random_state
from caussim.config import *

import seaborn as sns

from caussim.data.causal_df import CausalDf
from caussim.reports import save_figure_to_folders

# needed for latex functions such as \mathbb
plt.rcParams.update(
    {
        "font.size": 20,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
    }
)
import numpy as np

from caussim.estimation.estimation import get_selection_metrics, extrapolation_plot
from caussim.demos.utils import show_estimates, show_full_sample

from scipy.stats import multivariate_normal

random_seed = 0
generator = check_random_state(0)

# ### DGP ### #
N = 10000
pi_1 = 0.1
pi_0 = 1 - pi_1
mean_1 = -2
sigma_1 = 2
mean_0_0 = -2
sigma_0_0 = 2
mean_0_1 = 2
sigma_0_1 = 1

p_mode_0 = 0.5
step_constant = 10

dir2save = DIR2FIGURES / "r_risk_bound_example"
dir2save.mkdir(exist_ok=True, parents=True)

def p_1_density(x):
    return multivariate_normal.pdf(x, mean=mean_1, cov=sigma_1)


def p_0_density(x):
    return p_mode_0 * multivariate_normal.pdf(x, mean=mean_0_0, cov=sigma_0_0) + (
        1 - p_mode_0
    ) * multivariate_normal.pdf(x, mean=mean_0_1, cov=sigma_0_1)


def p_x_density(x):
    return pi_1 * p_1_density(x) + (1 - pi_1) * p_0_density(x)


x = np.linspace(-5, 3, N)
e = pi_1 * p_1_density(x) / p_x_density(x)
# compute treatment attribution
a = generator.binomial(n=1, p=e)
sample_df = pd.DataFrame({"a": a, "X_0": x, "e": e})

# Worst propensity score in our dataset among ALL samples (treated or not)
eta = np.min(sample_df["e"])

# Distribution of covariates
sns.displot(
    aspect=1.4,
    data=sample_df,
    x="X_0",
    hue="a",
    palette=COLOR_MAPPING,
    common_norm=False,
    linestyle="--",
    kind="kde",
    rug=True,
)
plt.savefig(
    dir2save / "x_density.pdf" 
)
# Check that the propensity score is decreasing on the whole axis
print(f"eta: {eta}")
# %%
# TODO: eta is not analytacillay derived, it should be computed analytically with the e(x) formula toa void breaking the bound. 
eta_multiplicator = 200
x_cut = np.min(sample_df.loc[sample_df["e"] <= eta_multiplicator * eta]["X_0"].values)

# response surface
sample_df["mu_0"] = 0
sample_df["mu_1"] = 1 + (sample_df["X_0"] >= x_cut) * step_constant
sample_df["y_1"] = sample_df["mu_1"]
sample_df["y_0"] = sample_df["mu_0"]
sample_df["y"] = (
    sample_df["a"] * sample_df["y_1"] + (1 - sample_df["a"]) * sample_df["y_0"]
)

extrapolation_plot(
    y_0=sample_df["y_0"],
    y_1=sample_df["y_1"],
    ps=sample_df["e"],
    a=sample_df["a"],
)
plt.savefig(
    dir2save / "extrapolation_plot.pdf", bbox_inches="tight"
)
# addding model
# One simple candidate function
predictions_bad_tau_risk = pd.DataFrame(
    {
        "hat_mu_0": np.zeros(N), 
        "hat_mu_1": np.ones(N), 
        "check_e": 0.5 * np.ones(N)}
)

predictions_good_tau_risk = pd.DataFrame(
    {
        "hat_mu_0": np.zeros(N), 
        "hat_mu_1": np.ones(N)*(1+step_constant/2), 
        "check_e": 0.5 * np.ones(N)}
)

for i, predictions in enumerate([predictions_bad_tau_risk, predictions_good_tau_risk]): 
    predictions["hat_tau"] = predictions["hat_mu_1"] - predictions["hat_mu_0"]
    predictions["check_m"] = (predictions["hat_mu_1"] - predictions["hat_mu_0"]) / 2
    sample_df["hat_mu_1"] = predictions["hat_mu_1"]
    sample_df["hat_mu_0"] = predictions["hat_mu_0"]
    sample_dataset = CausalDf(sample_df)
    fig = plt.figure(figsize=(11.5, 8))
    
    legend=False if i==1 else True
    ax, _ = show_full_sample(
        sample_df,
        fig,
        show_hat_y=True,
        show_sample=True,
        legend=legend,
        common_norm=False,
        ylims=(-0.5, step_constant + 2),
        pretty_axes_locator=False,
    )

    fig.suptitle("Oracle response surfaces")
    fig.tight_layout()
    estimates = get_selection_metrics(sample_dataset, predictions)
    fontsize=35
    show_estimates(
        ax,
        estimates,
        tau=False,
        tau_risk=True,
        oracle_r_risk=True,
        oracle_r_risk_ipw=True,
        #oracle_r_risk_rewritten=True,
        oracle_mu_iptw_risk=True,
        mu_risk=True,
        fontsize=fontsize,
        x_anchor=1.02,
        y_anchor=-0.4
    )
    bound_r_risk_txt = (
        "Bound on $R\mathrm{-risk}$:\n"
        + r"$R\mathrm{-risk}^*(\hat f)$="
        + f"{estimates['oracle_r_risk']:.4f}"
        + r"$\leq$"
        + f"\n{eta_multiplicator:.2e}"
        + r"$\eta \tau \mathrm{-risk}(\hat f)$="
        + f"{eta_multiplicator*eta*estimates['tau_risk']:.4f}"
        + r"""
    with $\eta=$"""
        + f"{eta:.2e}"
    )

    ax.text(
        1.02,
        0.99,
        bound_r_risk_txt,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgrey", alpha=0.5),
    )
    if i == 1:
        ax.set(ylabel="")
        plt.setp(ax.get_xticklabels(), visible=False)

    save_figure_to_folders(
        dir2save / f"r_risk_failure_{eta:.2e}_{eta_multiplicator:.2e}_y_estimator_{i}", figure_dir=True, paper_dir=True
    )
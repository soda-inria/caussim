#%%
# -*- coding: utf-8 -*-
# +
from matplotlib.pyplot import legend
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from caussim.config import *
from caussim.data.causal_df import CausalDf

# needed for latex functions such as \mathbb
plt.rcParams.update(
    {
        "font.size": 20,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
    }
)
import numpy as np

from caussim.data.simulations import sample_sigmoids
from caussim.demos.utils import show_estimates, show_full_sample, show_outcome_fit

RANDOM_STATE = 0
X_NOISE = 0
Y_NOISE = 0.01
N = 300
ALPHA_TREATED = 1
SCALE_TREATED = 0.6
ALPHA_UNTREATED = 2
SCALE_UNTREATED = 0.9
ALPHA_INTERVENTION = 0.8
TREATED_OFFSET = 0.1
MAX_OVERLAP = 0.95
CLIP = 1e-2

population_df = sample_sigmoids(
    n=N,
    alpha_treated=ALPHA_TREATED,
    scale_treated=SCALE_TREATED,
    alpha_untreated=ALPHA_UNTREATED,
    scale_untreated=SCALE_UNTREATED,
    treated_offset=TREATED_OFFSET,
    alpha_intervention=ALPHA_INTERVENTION,
    xlim=(0, 20),
    x_noise=X_NOISE,
    y_noise=Y_NOISE,
    random_state=RANDOM_STATE,
    max_overlap=MAX_OVERLAP,
)

sns.set_context("talk")

# For stepwise explanation
fig = plt.figure()
ax, _ = show_full_sample(
    population_df, fig, show_sample=False, legend=True, show_tau=False
)
fig.suptitle("Oracle response surfaces")
fig.tight_layout()
fig.savefig(
    str(DIR2FIGURES / f"oracle_mu.png"),
    bbox_inches="tight",
)
fig.savefig(
    str(DIR2FIGURES / f"oracle_mu.pdf"),
    bbox_inches="tight",
)
#%%

# Wo DM
fig = plt.figure()
ax, _ = show_full_sample(population_df, fig, legend=True)
causal_df = CausalDf(population_df)
true_estimates = causal_df.estimate_oracles()
show_estimates(ax, true_estimates, tau_DM=False)
fig.suptitle("Sampled population")
fig.tight_layout()
fig.savefig(
    str(DIR2FIGURES / f"sample.png"),
    bbox_inches="tight",
)
fig.savefig(
    str(DIR2FIGURES / f"sample.pdf"),
    bbox_inches="tight",
)

# With DM
fig = plt.figure()
ax, _ = show_full_sample(population_df, fig, show_DM=True, legend=True)
true_estimates = causal_df.estimate_oracles()
show_estimates(ax, true_estimates)
fig.suptitle("Sampled population")
fig.tight_layout()
fig.savefig(
    str(DIR2FIGURES / f"sample_w_DM.png"),
    bbox_inches="tight",
)
fig.savefig(
    str(DIR2FIGURES / f"sample_w_DM.pdf"),
    bbox_inches="tight",
)
# %%
# Find counter-example for R2 score on Y which is not the best causal model for abs bias ATE or tau_risk
paper_figsize = (12, 6)
fig = plt.figure(figsize=paper_figsize)
ax, _ = show_full_sample(population_df, fig, show_sample=True, legend=True)
fig.tight_layout()
fig.savefig(
    str(DIR2PAPER_IMG / f"oracle_mu.png"),
    bbox_inches="tight",
)
# %%
xanchor_estimates = 0.55
xanchor_metrics = 0.75
yanchor = 0.05
legend_size = 22

# good estimator (high r2 and small bias)
outcome_model = RandomForestRegressor(
    n_estimators=1, min_samples_leaf=8, random_state=0
)
ps_model = "linear"
# fig = plt.figure(constrained_layout=True, figsize=(12, 12))
# subfigs = fig.subfigures(2, 1, wspace=0.07)
fig = plt.figure(figsize=paper_figsize)
estimator, estimations, metrics, ax, ax_histx = show_outcome_fit(
    population_df,
    fig,
    outcome_model=outcome_model,
    propensity_model=ps_model,
    show_hat_y=True,
    clip=CLIP,
    n_splits=3,
    legend_prop={"size": legend_size},
)
show_estimates(
    ax,
    estimations,
    tau_G=True,
    x_anchor=xanchor_estimates,
    y_anchor=yanchor,
)
show_estimates(
    ax,
    estimations,
    tau=False,
    tau_risk=True,
    abs_bias_ATE=True,
    outcome_r2=True,
    metrics=metrics,
    x_anchor=xanchor_metrics,
    y_anchor=yanchor,
)
fig.tight_layout()
fig.savefig(
    str(DIR2PAPER_IMG / f"toy_random_forest_high_R2_high_tau_risk.png"),
    bbox_inches="tight",
)
fig.savefig(
    str(DIR2PAPER_IMG / f"toy_random_forest_high_R2_high_tau_risk.pdf"),
    bbox_inches="tight",
)
# %%
# small R2, small tau_risk
# outcome_model = RandomForestRegressor(n_estimators=1, random_state=0, min_samples_leaf=15)
fig = plt.figure(figsize=paper_figsize)
outcome_model = "linear_tlearn"
ps_model = "linear"
estimator, estimations, metrics, ax, ax_histx = show_outcome_fit(
    population_df,
    fig,
    outcome_model=outcome_model,
    propensity_model=ps_model,
    show_hat_y=True,
    clip=CLIP,
    n_splits=1,
    legend_prop={"size": legend_size},
)
show_estimates(
    ax,
    estimations,
    tau_G=True,
    x_anchor=xanchor_estimates,
    y_anchor=yanchor,
)
show_estimates(
    ax,
    estimations,
    tau=False,
    tau_risk=True,
    abs_bias_ATE=True,
    outcome_r2=True,
    metrics=metrics,
    x_anchor=xanchor_metrics,
    y_anchor=yanchor,
)
# ax_histx.text(0.1, 0.005, "covariates_distrib_label", fontweight="heavy")
fig.tight_layout()
fig.savefig(
    str(DIR2PAPER_IMG / f"toy_tlinear_model_small_R2_small_tau_risk.png"),
    bbox_inches="tight",
)
fig.savefig(
    str(DIR2PAPER_IMG / f"toy_tlinear_model_small_R2_small_tau_risk.pdf"),
    bbox_inches="tight",
)

# %%
# Different g-formula examples
# 1 - good model = random forest
OUTCOME_MODELS = ["random_forest", "linear", "linear_tlearn"]
for outcome_model in OUTCOME_MODELS:
    fig = plt.figure()
    estimator, estimations, metrics, ax, ax_histx = show_outcome_fit(
        population_df,
        fig,
        outcome_model=outcome_model,
        propensity_model=outcome_model,
        show_hat_y=True,
        clip=CLIP,
        n_splits=5,
    )
    show_estimates(ax, estimations, tau_G=True, tau_risk=True)
    fig.tight_layout()
    fig.savefig(
        str(DIR2FIGURES / f"{outcome_model}.png"),
        bbox_inches="tight",
    )


# IPW model
PROPENSITY_MODELS = ["linear", "random_forest"]
for ps_model in PROPENSITY_MODELS:
    fig = plt.figure()
    estimator, estimations, metrics, ax = show_outcome_fit(
        population_df,
        fig,
        outcome_model=ps_model,
        propensity_model=ps_model,
        show_e_oracle=True,
        show_mu_oracle=False,
        show_hat_y=False,
        show_hat_e=True,
        clip=CLIP,
        n_splits=5,
    )
    show_estimates(ax, estimations, tau_IPW=True, tau_IPW_oracle=True)
    fig.tight_layout()
    fig.savefig(
        str(DIR2FIGURES / f"ps_{ps_model}.png"),
        bbox_inches="tight",
    )

# %%

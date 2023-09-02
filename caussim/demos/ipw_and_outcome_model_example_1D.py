# %%
from matplotlib.pyplot import legend
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from caussim.config import *
from caussim.data.causal_df import CausalDf
from caussim.demos.utils import show_estimates, show_outcome_fit

import numpy as np

from caussim.data.simulations import sample_sigmoids
from caussim.demos.utils import show_estimates, show_full_sample, show_outcome_fit

# needed for latex functions such as \mathbb
plt.rcParams.update(
    {
        "font.size": 20,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
    }
)
# %%
RANDOM_STATE = 0
X_NOISE = 0
Y_NOISE = 0.1
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
# %%
# Different g-formula examples
# 1 - good model = random forest
outcome_model, ps_model = [Pipeline(
        [
            ("interaction", PolynomialFeatures(interaction_only=True)),
            ("reg", LinearRegression()),
        ]
    ), 
    Pipeline(
        [
            ("interaction", PolynomialFeatures(interaction_only=True)),
            ("reg", LogisticRegression()),
        ]
    )
    ]
outcome_model, ps_model = [
    RandomForestRegressor(random_state=10, n_estimators=10), 
    DecisionTreeClassifier(max_depth=5)
]
fig = plt.figure(figsize=(9, 6))
estimator, all_estimates, test_metrics, ax, ax_histx = show_outcome_fit(
    population_df,
    fig,
    outcome_model=outcome_model,
    propensity_model=ps_model,
    show_hat_y=True,
    clip=np.CLIP,
    n_splits=5,
)
#show_estimates(ax, all_estimates, tau_G=True, tau_risk=True)
fig.tight_layout()
fig.savefig(
    str(DIR2FIGURES / f"outcome_model__{outcome_model}.png"),
    bbox_inches="tight",
)
fig.savefig(
    str(DIR2FIGURES / f"outcome_model__{outcome_model}.pdf"),
    bbox_inches="tight",
)

#%%
# IPW model
ps_model = RandomForestClassifier(random_state=10, n_estimators=10)
fig = plt.figure(figsize=(9, 6))
estimator, all_estimates, test_metrics, ax, ax_histx = show_outcome_fit(
    population_df,
    fig,
    outcome_model=outcome_model,
    propensity_model=ps_model,
    show_e_oracle=True,
    show_mu_oracle=False,
    show_hat_y=False,
    show_hat_e=True,
    clip=CLIP,
    n_splits=5,
)
#show_estimates(ax, all_estimates, tau_IPW=True, tau_IPW_oracle=True)
fig.tight_layout()
fig.savefig(
    str(DIR2FIGURES / f"ipw_model__{ps_model}.pdf"),
    bbox_inches="tight",
)
fig.savefig(
    str(DIR2FIGURES / f"ipw_model__{ps_model}.png"),
    bbox_inches="tight",
)
# %%

# %%
from sklearn.model_selection import ParameterGrid
from caussim.data.loading import load_dataset, make_dataset_config

from sklearn.utils import check_random_state
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from caussim.demos.utils import plot_simu1D_cut, plot_simu_1D_cuts, plot_simu_2D_1D_cuts
from caussim.pdistances.effect_size import mean_causal_effect

RANDOM_STATE = 0
generator = check_random_state(RANDOM_STATE)
# %%
# Q: 
# 1- Do we explore a lot of causal effects ratio ? 
# 2 - What parameters do I need to set to change the causal effect ratios ? 
# I:
# Yes 
# The parameters to set is the effect_size.
# A:
# The effect size from 0 to 1 (excluded) is a good parameter knot to control the causal effect ratio as measured by \Delta_{\mu} implemented by mean_causal_effect. 
# 1 -
# %% 
xp_grid = {
        "dataset_name": ["caussim"],
        "overlap": generator.uniform(0, 2.5, size=100),
        "random_state": list(range(1, 4)),
        "treatment_ratio": [0.25, 0.5, 0.75],
    }

delta_mu_of_x_grid = []
for dataset_setup in tqdm(ParameterGrid(xp_grid)):
    dataset_config = make_dataset_config(**dataset_setup)
    sim, dataset = load_dataset(dataset_config=dataset_config)
    delta_mu_of_x_grid.append(
    
        mean_causal_effect(dataset.df.mu_1, dataset.df.mu_0)
    )
# %%
delta_mu_of_x_grid_ = np.array(delta_mu_of_x_grid)
delta_mu_of_x_grid_wo_xtrm = delta_mu_of_x_grid_[np.abs(delta_mu_of_x_grid_) < 100]
print(pd.DataFrame(delta_mu_of_x_grid).describe(
    percentiles=np.array([1, 10, 25, 50, 60, 65, 70, 75, 90, 99])/100).T.to_markdown(index=False)
    )

plt.hist(delta_mu_of_x_grid_wo_xtrm, bins=100)
plt.xlim(0, 50)
# %%
# 2 - 
# %%
dataset_setup = {
    "dataset_name": "caussim",
    "overlap": 1,
    "effect_size": 0.2,
    "random_state": 10,
    "treatment_ratio": 0.5,
}

for effect_size in [0.01, 0.1, 0.2, 0.5, 0.7, 0.8, 0.99]:
    dataset_setup["effect_size"] = effect_size
    dataset_config = make_dataset_config(**dataset_setup)
    sim, dataset = load_dataset(dataset_config=dataset_config)

    delta_mu = mean_causal_effect(dataset.df.mu_1, dataset.df.mu_0)
    print(f"Effect size parameter: {effect_size} vs delta_mu {delta_mu}" )
    _, _ = plot_simu_1D_cuts(dataset.df, sim)

# %% 
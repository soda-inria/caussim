# %%
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
import yaml
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state
from tqdm import tqdm
from caussim.config import DIR2EXPES
from copy import deepcopy

from caussim.reports.utils import get_expe_indices
from caussim.data.loading import load_dataset, make_dataset_config
from caussim.pdistances.effect_size import mean_causal_effect
import numpy as np
# %%
xp_name = "caussim__nuisance_non_linear__candidates_ridge__overlap_01-247_separated_nuisance_train_set"
xp_name = "caussim__nuisance_linear__candidates_ridge__overlap_05-247"
caussim_logs = pd.read_csv(DIR2EXPES/f"caussim/{xp_name}/run_logs.csv")
caussim_logs.head()
# %
overlap_param, expe_indices = get_expe_indices(caussim_logs)

unique_indices = caussim_logs[expe_indices].drop_duplicates()
print(unique_indices)

# %% 
# I need to rerun the exact same simulations and make the join on the overlap parameter and test seed.  
RANDOM_STATE = 0
generator = check_random_state(RANDOM_STATE)

dataset_grid = {
        "dataset_name": ["caussim"],
        "overlap": generator.uniform(0, 2.5, size=25),
        "random_state": list(range(1, 4)),
        "treatment_ratio": [0.25, 0.5, 0.75],
        "effect_size": [0.1, 0.5, 0.9],
    }
dataset_name = "caussim"
delta_mu_distrib = []
for dataset_setup in tqdm(list(ParameterGrid(dataset_grid))):
    dataset_config = make_dataset_config(**dataset_setup)
    sim, dgp_sample = load_dataset(dataset_config=dataset_config)
    delta_mu = mean_causal_effect(dgp_sample.df.mu_1, dgp_sample.df.mu_0)
    delta_mu_distrib.append(delta_mu)
# %%
print(pd.DataFrame(delta_mu_distrib).describe(
    percentiles=np.array([1, 10, 25, 50, 60, 65, 70, 75, 90, 99])/100).T.to_markdown(index=False)
    )
plt.hist(delta_mu_distrib, bins=100)
plt.xlim(0, 50)
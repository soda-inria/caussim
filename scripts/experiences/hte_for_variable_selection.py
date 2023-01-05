# %% 
from copy import deepcopy
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from caussim.config import DIR2EXPES, DIR2REPORTS, SCRIPT_NAME
from caussim.data.simulations import sample_wager
from caussim.estimation.estimation import CaussimLogger
from caussim.estimation.estimators import AteEstimator
from tqdm import tqdm
"""
Does more treatment heterogeneity makes disappear the bias obtained when using outcome-only variables ? 

Expe: Learn different ate models with RF and varying number of samples. Make repeated sample seeds to have a variance. 

Make a plot for 3 values of HTE : 0, 0.5, 1

HYP: Bias should be 
"""
# %% 
dir2expe_results = DIR2EXPES / SCRIPT_NAME
dir2expe_results.mkdir(exist_ok=True, parents=True)
causal_logger = CaussimLogger()
# %%

df_rs=0
df_setup = "A"
model_rs = 0
N_BS = 10

n_samples = [100, 300, 1000, 3000]#, 10000, 30000]
treatment_heterogeneities = [0, 0.5, 1]
outcome_model = RandomForestRegressor(random_state=model_rs)
propensity_model = RandomForestClassifier(random_state=model_rs)

for n in tqdm(n_samples):
    for hte in treatment_heterogeneities:
        covariates_sets = {
            "extended": [0, 1, 2, 3, 4],
            "minimal": [0, 1],
            "smart": [0, 1]
        }
        if hte > 0:
            covariates_sets["smart"] += [2]
        for covariate_subset in covariates_sets.items():

            true_causal_df = sample_wager(
                n=n,
                setup="A", 
                hte=hte, 
                random_state=df_rs
            )
            true_ate = true_causal_df.estimate_oracles()["ate"]
            # make some bootstrap
            for seed in np.arange(N_BS):
                causal_df = deepcopy(true_causal_df)
                causal_df.bootstrap(seed)
                
                causal_model = AteEstimator(
                    outcome_model=outcome_model,
                    propensity_model=propensity_model,
                    tau="AIPW",
                    random_state_cv=model_rs,
                )
                predictions, estimates, metrics = causal_model.fit_predict_estimate(
                    X=causal_df.get_aX(),
                    y=causal_df.get_y(),
                )
                causal_logger.log_simulation_result(
                    causal_df=causal_df,
                    causal_estimator=causal_model,
                    estimates=estimates,
                    parameters={"n_samples": n, "subset": covariate_subset, "hte":hte, "model_rs":model_rs, "true_ate": true_ate}
                    )
# %%
causal_logger.to_pandas().to_csv(
    dir2expe_results / f"dataset_{causal_df.dataset_name}__setup_{df_setup}__rs_{df_rs}__hte_{treatment_heterogeneities}__y_{type(outcome_model).__name__}__a_{type(propensity_model).__name__}.csv", index=False
)

# %%
import pandas as pd
import seaborn as sns

from caussim.config import DIR2EXPES, DIR2NOTES, SCRIPT_NAME, TAB_COLORS
from caussim.report_utils import plot_diff_between_n_tv_and_approximation

sns.set(font_scale=1.2, style="whitegrid", context="talk")
# %%
dir2save = (DIR2NOTES / "n_tv_approximation")
dir2save.mkdir(exist_ok=True, parents=True)
dir2results = DIR2EXPES / "normalized_total_variation_approximation"
n_tv_results = pd.concat([pd.read_csv(f) for f in list(dir2results.iterdir())], axis=0)

# %% 
n_tv_results_long = pd.wide_to_long(
    n_tv_results,
    i=set(n_tv_results.columns).difference(['n_tv_wo_calibration',
       'bss_wo_calibration', 'bs_wo_calibration', 'roc_auc_wo_calibration',
       'n_tv_calibrated', 'bss_calibrated', 'bs_calibrated',
       'roc_auc_calibrated']),
    stubnames = ["n_tv", "bss", "bs", "roc_auc"],
    suffix=r"\w+",
    sep="_",
    j="calibration"
).reset_index()
# %%
# Most important plot
metric = "oracle_n_tv"
calibration = "calibrated"
n_tv_results_c = n_tv_results_long[n_tv_results_long['calibration'] == calibration]
plot_diff_between_n_tv_and_approximation(n_tv_results=n_tv_results_c, x_colname=metric, calibration=calibration=="calibrated")

# %% 
# All plots

for metric in ["oracle_n_tv", "bss", "bs","roc_auc"]:
    for calibration in ["wo_calibration", "calibrated"]:
            n_tv_results_c = n_tv_results_long[n_tv_results_long['calibration'] == calibration]
            plot_diff_between_n_tv_and_approximation(n_tv_results=n_tv_results_c, x_colname=metric, calibration=calibration=="calibrated")


# %%

# %%
"""
This file gives some empirical arguments supporting that Normalized Total Variation is approximating the overlap between populations.
"""
from matplotlib.pyplot import savefig
import pandas as pd
import numpy as np
import re
from caussim.reports import read_logs
from caussim.config import *
from caussim.reports.plots_utils import METRIC_OF_INTEREST_LABELS
from caussim.reports.utils import save_figure_to_folders

pd.set_option("display.max_columns", None)
# %%

# MMD and NTV correlates well for Caussim (gaussian distribution of population covariates)
xp_name = Path(
    DIR2EXPES
    / "caussim_save"
    / "caussim__stacked_regressor__test_size_5000__n_datasets_1000"
)
run_logs, simu_config = read_logs(xp_name)
dataset_name = run_logs["dataset_name"].values[0]

x_overlap_measure = "test_transformed_mmd"
y_overlap_measure = "test_d_normalized_tv"
unique_setups = run_logs[[x_overlap_measure, y_overlap_measure]].drop_duplicates()
logging.info("Nb unique setups: {}".format(unique_setups.shape))
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
sns.regplot(
    ax=ax,
    data=unique_setups,
    x=x_overlap_measure,
    y=y_overlap_measure,
    lowess=True,
    scatter_kws={"alpha": 0.5},
)
ax.set(
    xlabel=METRIC_OF_INTEREST_LABELS[x_overlap_measure],
    ylabel=METRIC_OF_INTEREST_LABELS[y_overlap_measure],
)
save_figure_to_folders(
    figure_name=Path("overlap_measure") / f"{dataset_name}_transformed_mmd_vs_ntv",
    figure_dir=True,
    paper_dir=True,
)
# %%
overlap_measure = "test_d_normalized_tv"
overlap_measure_label = METRIC_OF_INTEREST_LABELS[overlap_measure]
# Recovery of penalized setups for ACIC 2016
settings_acic16 = pd.read_csv(
    DIR2SEMI_SIMULATED_DATA / "acic_2016" / "simulation_setups.csv"
)
overlap_dic = {"full": TAB_COLORS[0], "penalize": TAB_COLORS[2]}
path2dataset_ps = DIR2REPORTS / f"acic_2016_ps.pkl"
all_ps = pd.read_pickle(path2dataset_ps)
all_ps = all_ps.merge(settings_acic16, on="dgp")
# %%
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
g = sns.scatterplot(
    ax=ax,
    data=all_ps,
    x="nTV",
    y="dgp",
    hue="overlap",
    palette={"penalize": TAB_COLORS[8], "full": TAB_COLORS[1]},
    s=300,
)
ax.set(xlabel=overlap_measure_label, ylabel="Simulation generation process ID")
g.get_legend().set(title="Overlap setup")
plt.setp(ax.get_legend().get_texts(), fontsize="32")  # for legend text
plt.setp(ax.get_legend().get_title(), fontsize="40")  # for legend title

save_figure_to_folders(
    figure_name=Path("overlap_measure") / f"acic_2016_recovery_overlap_setup",
    figure_dir=True,
    paper_dir=True,
)
# %%
# ### Distribution of weights for the vs MMD ### #
xps = [
    # For exploration of seed effects on this plot
    Path(
        DIR2EXPES
        / "caussim_save"
        / "caussim__nuisance_non_linear__candidates_ridge__overlap_0012-247_balanced_populations"
    ),
    # All 4 datasets plot
    # Path(DIR2EXPES / "acic_2018_save" / "acic_2018__non_linear__first_uid_1000"),
    Path(
        DIR2EXPES
        / "caussim_save"
        / "caussim__stacked_regressor__test_size_5000__n_datasets_1000"
    ),
    # Path(DIR2EXPES / "acic_2016_save"/ "acic_2016__stacked_regressor__dgp_1-77__seed_1-10"),
    # Path(DIR2EXPES / "twins_save" / "twins__stacked_regressor__rs_1-10__overlap_0.1-3"),
]
ipw_weights_labels = {
    "oracle_IPW_max": "$max(IPW^*)$",
    "IPW_max": "$max(\widehat{IPW})$",
}
for xp_name in xps:
    expe_results, _ = read_logs(xp_name)
    ipw_weights_colors = {
        ipw_weights_labels["oracle_IPW_max"]: TAB_COLORS[2],
        ipw_weights_labels["IPW_max"]: TAB_COLORS[4],
    }
    dataset_name = expe_results["dataset_name"].values[0]
    if dataset_name == "acic_2018":
        ipw_weights_labels.pop("oracle_IPW_max")
        overlap_measure = "hat_d_normalized_tv"
    else:
        overlap_measure = "test_d_normalized_tv"
    overlap_measure_label = METRIC_OF_INTEREST_LABELS[overlap_measure]
    ipw_var_name = "Test set IPWs"
    ipw_value_name = r"$max(\frac{a}{e} + \frac{1-a}{1-e})$"
    ipw_weights = (
        expe_results[[*ipw_weights_labels.keys(), overlap_measure]]
        .drop_duplicates()
        .melt(
            id_vars=overlap_measure,
            var_name=ipw_var_name,
            value_name=ipw_value_name,
        )
    )
    ipw_weights[ipw_var_name] = ipw_weights[ipw_var_name].map(
        lambda x: ipw_weights_labels[x]
    )
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    sns.scatterplot(
        ax=ax,
        data=ipw_weights,
        x=overlap_measure,
        y=ipw_value_name,
        hue=ipw_var_name,
        palette=ipw_weights_colors,
    )
    plt.yscale("log")
    plt.xlabel(overlap_measure_label)

    save_figure_to_folders(
        figure_name=Path("overlap_measure") / f"{dataset_name}_max_ipw_vs_ntv",
        figure_dir=True,
        paper_dir=True,
    )
# %%
# fig, ax = plt.subplots(1, 1, figsize=(14, 10))
# sns.scatterplot(
#     ax=ax,
#     data=run_logs,
#     x=overlap_measure,
#     y="oracle_IPW_max",
# )
# # %%

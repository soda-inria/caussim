# %%
import pandas as pd
import numpy as np
import re
from caussim.reports.utils import (
    read_logs
)
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from caussim.reports.plots_utils import (
    CAUSAL_METRICS, METRIC_OF_INTEREST_LABELS, create_legend_for_candidates
)

from caussim.utils import *
from caussim.config import *

sns.set_style("whitegrid")

pd.set_option("display.max_columns", None)
# %%
# xp_name = ""acic_2016_save/2022-03-26-21-05-40_acic_scoring"  # stacked regressor and all candidates are hgb
xp_name = "acic_2016_ate_heterogeneity_save/2022-04-08-17-00-17_acic_2016_ate_heterogeneity"  # no nuisnaces and 6 candidates (ridge and hgb)
xp_path = Path(DIR2EXPES / xp_name)
run_logs, simu_config = read_logs(xp_path)

overlap_measure = "dgp_d_normalized_tv"
# overlap_measure = "test_d_normalized_tv"
run_logs = run_logs.rename(
    columns={"d_js": "test_d_js"}
)

xp_savename = (
    re.search("\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", xp_name).group(0)
    + f"__d_js={run_logs['test_d_js'].min().round(4)}_{run_logs['test_d_js'].max().round(4)}"
)

run_logs["r_risk_ipw_corrected"] = run_logs["r_risk_ipw"] + 2 * run_logs["r_risk"]
run_logs["oracle_r_risk_ipw_corrected"] = (
    run_logs["oracle_r_risk_ipw"] + 2 * run_logs["oracle_r_risk"]
)

CAUSAL_METRICS = [metric for metric in CAUSAL_METRICS if metric in run_logs.columns]


# nuisance_models_label = get_nuisances_type(run_logs)
model_keys = ["simulation_param", "simulation_seed", overlap_measure, "rs_test_split"]
model_estimations = ["ate", "bias_ate", "tau_risk", "r2"]
model_params = [
    "cate_candidate_model",
    "meta_learner_name",
    "final_estimator__learning_rate",
    # "final_estimator__max_leaf_nodes",
    "final_estimator__alpha",
]
# %%
simu_seed = 1
mask_simu_seed = run_logs["simulation_seed"] == simu_seed
simus_to_plots = run_logs.loc[
    mask_simu_seed, model_keys + model_params + model_estimations
].sort_values(overlap_measure)

simus_to_plots["candidate_id"] = simus_to_plots.apply(
    lambda x: f"{x['cate_candidate_model']}__"
    + "__".join([f"{p}_{str(x[p])}" for p in model_params[1:]]),
    axis=1,
)
simus_to_plots["abs_bias_ate"] = np.abs(simus_to_plots["bias_ate"])
# %%
candidates_family = np.sort(simus_to_plots["candidate_id"].unique())
# resort for better colors
colormap = [
    cm.tab20(1) ,
    cm.tab20(0) ,
    cm.tab20(2) ,
    cm.tab20(3) ,
    cm.tab20(4) ,
    cm.tab20(5) ,
    ]

(handles, labels), candidates_colormap = create_legend_for_candidates(candidates_family, colormap=colormap)
print(labels)

# %%
cap_size = 10
fig, ax = plt.subplots(1, 1, figsize = (14, 7))
for candidate in simus_to_plots["candidate_id"].unique():
    candidate_data = simus_to_plots.loc[simus_to_plots["candidate_id"] == candidate, :]
    #candidate_data["hat_ate"] = candidate_data["bias_ate"] + candidate_data["ate"]
    sns.lineplot(
        ax=ax,
        data=candidate_data,
        x=overlap_measure,
        y="abs_bias_ate",
        color=candidates_colormap[candidate],
        marker="o",
        #ci="sd",
        linestyle="",
        err_style="bars",
        err_kws={"capsize": cap_size},
        legend=False,
    )
ax.set_xlabel(METRIC_OF_INTEREST_LABELS[overlap_measure])
log_scale = True
ylabel = "Absolute bias to the true ATE"
if log_scale:
    ax.set_yscale("log")
    ylabel += "\n log scale"
ax.set_ylabel(ylabel)
ax.annotate(xy=(-0.1,0.9), text="Worse", fontsize=25, xycoords="axes fraction", annotation_clip=False, color="red")
ax.annotate(xy=(-0.1,0.1), text="Better", fontsize=25, xycoords="axes fraction", annotation_clip=False, color="green")
# # add true value of ATE

# gold_ates = simus_to_plots.loc[
#     :, ["simulation_param", overlap_measure, "ate"]
# ].drop_duplicates()
# gold_ate_color = "black"
# gold_ate_marker = "o"
# sns.lineplot(
#         ax=ax,
#         data=gold_ates,
#         x=overlap_measure,
#         y="ate",
#         color=gold_ate_color,
#         marker=gold_ate_marker,
#         #ci="sd",
#         linestyle="",
#         err_style="bars",
#         err_kws={"capsize": cap_size},
#         legend=False,
#     )

# ax.scatter(
#     x=gold_ates[overlap_measure],
#     y=gold_ates["ate"],
#     c=gold_ate_color,
#     marker=gold_ate_marker,
#     s=50,
# )
# gold_ate_handme = Line2D(
#     [0],
#     [0],
#     color=gold_ate_color,
#     marker=gold_ate_marker,
#     markersize=12,
#     linestyle="none",
# )
# gold_ate_label = "True ATE"
plt.rc('legend',fontsize=20) # using a size in points
#plt.add_artist(
plt.legend(
    handles,#, gold_ate_handme],
    labels,#, gold_ate_label],
    bbox_to_anchor=(0, 1.05, 1, 0), loc="lower left", mode="expand",ncol=2,#(0.01, 1.01),
    title="Outcome models",
    borderaxespad=0,
    prop={'size': 18}
)

figname = f"{xp_path.name}_abs_bias_ylog_scale={log_scale}"
fig.savefig(
            DIR2FIGURES
            / (figname + ".pdf"),
            bbox_inches="tight",
        )
fig.savefig(
            DIR2PAPER_IMG
            / (figname + ".pdf"),
            bbox_inches="tight",
        )

# %%

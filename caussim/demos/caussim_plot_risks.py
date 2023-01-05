# %%
from sklearn import clone
from caussim.data.loading import load_dataset
from caussim.estimation.estimators import (
    SFTLEARNER_LABEL,
    SLEARNER_LABEL,
    TLEARNER_LABEL,
    CateEstimator,
    IdentityTransformer,
    set_meta_learner,
)
import numpy as np
from copy import deepcopy
from caussim.config import *
from caussim.estimation.estimation import get_selection_metrics

from caussim.demos.utils import (
    plot_mu_risk,
    plot_simu2D,
    plot_tau_risk,
)
from caussim.experiences.base_config import DEFAULT_SIMU_CONFIG, CATE_CONFIG_LOGISTIC_NUISANCE
from caussim.reports.utils import save_figure_to_folders

dataset_config = deepcopy(DEFAULT_SIMU_CONFIG)
cate_config = deepcopy(CATE_CONFIG_LOGISTIC_NUISANCE)
dataset_config["simu_random_seed"] = 0
base_seed = 10
cate_config["rs_prefit"] = 2
overlap = 2.5
n_components = 2
dataset_config["treatment_ratio"] = 0.2
dataset_config["treatment_link"]["params"]["random_state_feat"] = base_seed
dataset_config["baseline_link"]["params"]["random_state_feat"] = base_seed
dataset_config["outcome_noise"] = 0
dataset_config["treatment_link"]["params"]["n_components"] = n_components
dataset_config["baseline_link"]["params"]["n_components"] = n_components
dataset_config["treatment_assignment"]["params"]["overlap"] = overlap
n_samples = 1000
for set in ["test", "validate", "train"]:
    dataset_config[f"{set}_size"] = n_samples

sim, causal_df = load_dataset(dataset_config=dataset_config)
# plot_overlap(
#     df,
#     treatment_assignment=sim.treatment_assignment_type,
#     overlap=sim.overlap,
#     random_seed=sim.random_seed,
# )
_, _ = plot_simu2D(causal_df.df, sim)
# %%
# cate_config["rs_learner"] = 0

quantile = 0.6
# cate_config.update(**{"n_splits": 1})
for learner in [TLEARNER_LABEL]:
    cate_config["meta_learner_name"] = learner
    sim.rs_gaussian = dataset_config["train_seed"]
    df_train = sim.sample(num_samples=dataset_config["train_size"]).df
    sim.rs_gaussian = dataset_config["test_seed"]
    df_test = sim.sample(
        num_samples=dataset_config["test_size"],
    ).df
    featurizer = clone(
    sim.baseline_pipeline.pipeline.named_steps.get(
        "featurization", IdentityTransformer()
    )
    )
    # Force nuisance featurizer to be the same as the one of the test dataset. 
    # Finally we have :
    # - a train_featurizer used only for train_df generation, 
    #  - a test featurizert used for test_df generation, nuisance estimators, candidate estimators. 
    y_featurizer_name = "featurizer"
    a_featurizer_name = "featurizer"
    if cate_config["y_estimator"].get_params().get("estimators") is not None:
        y_featurizer_name = "featurized_ridge__" + y_featurizer_name
        a_featurizer_name = "featurized_logisticregression__" + a_featurizer_name
    cate_config["y_estimator"] = cate_config["y_estimator"].set_params(
        **{y_featurizer_name: featurizer}
    )
    cate_config["a_estimator"] = cate_config["a_estimator"].set_params(
        **{a_featurizer_name: featurizer}
    )
    base_meta_learner = set_meta_learner(
        SLEARNER_LABEL, final_estimator=cate_config["final_estimator"], featurizer=featurizer
    )
    cate_estimator = CateEstimator(
        meta_learner=base_meta_learner,
        a_estimator=cate_config["a_estimator"],
        y_estimator=cate_config["y_estimator"],
        a_hyperparameters=cate_config["a_hyperparameters"],
        y_hyperparameters=cate_config["y_hyperparameters"],
        a_scoring=cate_config["a_scoring"],
        y_scoring=cate_config["y_scoring"],
        n_iter=cate_config["n_iter_random_search"],
        cv=cate_config["n_splits"],
        random_state_hp_search=cate_config["rs_hp_search"],
    )
    X_train, y_train = df_train.get_aX_y()
    cate_estimator.fit(X_train, y_train)
    X_test, y_test = df_test.get_aX_y()
    test_predictions = cate_estimator.predict(X=X_test)
    metrics = get_selection_metrics(df_test, test_predictions)
    info = df_test.describe()
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig, _ = plot_mu_risk(
        df_test,
        sim,
        cate_estimator,
        test_predictions,
        metrics,
        max_quantile=quantile,
        fig=fig,
        ax=axes[0, :],
    )
    _, _ = plot_mu_risk(
        df_test,
        sim,
        cate_estimator,
        test_predictions,
        metrics,
        max_quantile=quantile,
        risk="mu_iptw_oracle",
        fig=fig,
        ax=axes[1, :],
    )
    fig_tau, ax_tau = plot_tau_risk(
        df_test,
        sim,
        cate_estimator,
        test_predictions,
        max_quantile=quantile,
        fig=fig,
        ax=axes[2, 0],
    )
    # axes[-1, -1].set_visible(False)
    _, _ = plot_simu2D(df_test, sim, fig=fig, ax=axes[-1, -1])
    axes[-1, -1].set_title("")
    axes[-1, -1].get_legend().remove()
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.suptitle(
        "$\\tau\mathrm{-residuals}$ map is closer from the superposition of \n reweighted $\mu\mathrm{-residuals}$ maps than from $\mu\mathrm{-residuals}$"
    )
    save_figure_to_folders(figure_name="caussim_plot_risk", figure_dir=True)
# %%

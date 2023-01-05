from copy import deepcopy
from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.model_selection import (
    train_test_split,
    ParameterGrid,
)

from caussim.config import DIR2EXPES, DIR2FIGURES
from caussim.data.causal_df import CausalDf
from caussim.data.loading import load_dataset
from caussim.demos.utils import (
    plot_metric_comparison,
    plot_models_comparison,
    show_estimates,
    show_full_sample,
)
from caussim.estimation.estimators import (
    SLEARNER_LABEL,
    CateEstimator,
    IdentityTransformer,
    set_meta_learner,
)

from caussim.estimation.estimation import (
    extrapolation_plot_outcome,
    extrapolation_plot_residuals,
    get_selection_metrics,
)
import pyarrow as pa
import pyarrow.parquet as pq
from caussim.experiences.pipelines import REGRESSORS

from caussim.reports.plots_utils import CAUSAL_METRIC_LABELS, CAUSAL_METRICS
from caussim.reports.utils import (
    get_metric_rankings_by_dataset,
    get_rankings_aggregate,
    save_figure_to_folders,
)
from caussim.utils import dic2str


def run_causal_scores_eval(
    dataset_config,
    cate_config,
    candidate_estimators_grid,
    xp_name=None,
    extrapolation_plot: bool = False,
    compute_rankings: bool = False,
    write_to_parquet: bool = False,
    test: bool = False,
    show_ntv: bool = True,
):
    """
    Score several g-formula models with mu_iptw_risk (reweighted mse) and mu_risk (mse on y) on different semi-simulated datasets"""
    dataset_name = dataset_config.get("dataset_name")
    expe_timestamp = datetime.now()

    path2results = DIR2EXPES / dataset_name / xp_name
    path2results.mkdir(exist_ok=True, parents=True)
    dir2extrapolation_plot = DIR2FIGURES / "extrapolation_plots" / xp_name
    if dir2extrapolation_plot.exists():
        shutil.rmtree(dir2extrapolation_plot)
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    )
    logger = logging.getLogger()
    logger.info(f"Logging figures at {dir2extrapolation_plot}")
    candidate_estimators = []
    for candidate_estimator_params in candidate_estimators_grid:
        candidate_estimators += ParameterGrid(candidate_estimator_params["hp_kwargs"])
    nb_runs = len(candidate_estimators)
    logger.info(
        f"""\n
    -----NEW EXPERIMENT-----
    Number of fitted candidate learners: {nb_runs}\n
    Data set config: {dataset_config}\n
    Candidate estimators grid: {candidate_estimators_grid}\n
    """
    )
    # Load a dgp
    sim, dgp_sample = load_dataset(dataset_config=dataset_config)
    # Only for testing
    if test is True:
        dgp_sample.df = dgp_sample.df.sample(n=100, random_state=0)
    if dataset_name != "caussim":
        df_nuisance_set, df_test_ = train_test_split(
            dgp_sample.df,
            test_size=cate_config["test_ratio"],
            random_state=cate_config["rs_test_split"],
        )
        if "nuisance_set_ratio" in cate_config.keys():
            df_train, df_test = train_test_split(
                df_test_,
                test_size=cate_config["nuisance_set_ratio"],
                random_state=cate_config["rs_nuisance_set_split"],
            )
        else:
            df_train = df_nuisance_set.copy()
            df_test = df_test_
        featurizer = None
    else:
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
        if "nuisance_set_size" in dataset_config.keys():
            sim.rs_gaussian = dataset_config["nuisance_set_seed"]
            df_nuisance_set = sim.sample(
                num_samples=dataset_config["nuisance_set_size"],
            ).df
        # In the case of caussim, I allow the user to specify a different set for training separately the nuisances.
        else:
            df_nuisance_set = df_train.copy()

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

    df_train = CausalDf(df_train.reset_index(drop=True))
    df_test = CausalDf(df_test.reset_index(drop=True))
    df_nuisance_set = CausalDf(df_nuisance_set.reset_index(drop=True))

    X_test, y_test = df_test.get_aX_y()
    X_train, y_train = df_train.get_aX_y()
    X_nuisance_set, y_nuisance_set = df_nuisance_set.get_aX_y()
    # This final estimator is not used, this is the candidate estimator that is important. We only need it to instantiate the CateEstimator. But we should be able to instiantiate it without it.
    base_meta_learner = set_meta_learner(
        SLEARNER_LABEL,
        final_estimator=cate_config["final_estimator"],
        featurizer=featurizer,
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
    logger.info(
        f"Prefit nuisances models on a nuisance set of shape {X_nuisance_set.shape}."
    )
    if ("nuisance_set_ratio" in cate_config.keys()) or (
        "nuisance_set_size" in dataset_config.keys()
    ):
        logger.info(f"The nuisance set differs from the train set for candidates.")

    cate_estimator.fit_nuisances(X=X_nuisance_set, y=y_nuisance_set)

    logger.info(
        f"Iterate over candidate estimators.\nFitted on train set of shape {X_train.shape}, inference on test set of shape {X_test.shape}"
    )

    best_metric_names = [
        "tau_risk",
        "oracle_mu_iptw_risk",
        "oracle_r_risk",
        "oracle_r_risk_IS2",
        "mu_risk",
    ]
    best_metrics = {m: np.inf for m in best_metric_names}
    best_models_id = {m: "" for m in best_metric_names}
    best_models_predictions = {m: pd.DataFrame() for m in best_metric_names}
    best_metric_params = {m: {} for m in best_metric_names}
    ##TODO: could be parameters
    metrics_to_compared = [
        ("tau_risk", "oracle_r_risk"),
        ("oracle_r_risk", "oracle_r_risk_IS2"),
        ("oracle_r_risk", "mu_risk"),
        ("oracle_r_risk", "oracle_mu_iptw_risk"),
    ]

    # Results for the whole run (at the (dataset x dgp x random seed) level)
    expe_results = []
    for candidate_model_params in candidate_estimators_grid:
        # Iterate over meta-learner parameters
        for parameter_grid in list(ParameterGrid(candidate_model_params["hp_kwargs"])):
            t0 = datetime.now()
            featurizer = candidate_model_params.get("featurizer", None)
            cate_estimator_learner = deepcopy(cate_estimator)
            candidate_meta_learner = set_meta_learner(
                meta_learner_name=candidate_model_params["meta_learner_name"],
                final_estimator=REGRESSORS[candidate_model_params["estimator"]],
                featurizer=featurizer,
            )

            candidate_meta_learner.set_params(**parameter_grid)
            cate_estimator_learner.meta_learner = candidate_meta_learner

            cate_estimator_learner.fit(X_train, y_train)
            # Cross validation for score predictions is available (ie reusing fitted model nuisances and cv associated): here useless because we are predicting on a different set.
            test_predictions = cate_estimator_learner.predict(X=X_test)

            metrics = get_selection_metrics(
                causal_df=df_test, predictions=test_predictions
            )
            df_validate_info = df_test.describe(prefix="test_")

            log_cate_config = deepcopy(cate_config)
            log_cate_config["y_estimator"] = str(log_cate_config["y_estimator"])
            log_cate_config["a_estimator"] = str(log_cate_config["a_estimator"])
            # logging results
            xp_param = {
                **dataset_config,
                **log_cate_config,
                **parameter_grid,
                **df_validate_info,
                **metrics,
                **{"cate_candidate_model": candidate_model_params["estimator"]},
                **{"meta_learner_name": candidate_model_params["meta_learner_name"]},
            }
            xp_param["compute_time"] = datetime.now() - t0
            run_id = hash(
                t0.strftime("%Y-%m-%d-%H-%M-%S")
                + json.dumps(parameter_grid)
                + json.dumps(metrics)
            )
            run_results = pd.DataFrame.from_dict(xp_param, orient="index").transpose()
            run_path = path2results / f"run_{run_id}.csv"
            run_results.to_csv(run_path, index=False)
            logger.info(f"Save candidate estimator result at {run_path}")
            if write_to_parquet:
                test_data_to_write = pd.concat([df_test.df, test_predictions], axis=1)
                test_data_to_write["run_id"] = run_id
                table = pa.Table.from_pandas(test_data_to_write)
                pq.write_to_dataset(
                    table, root_path=str(path2results / "test_data.parquet")
                )
            expe_results.append(run_results)
            if extrapolation_plot:
                dir2extrapolation_plot.mkdir(exist_ok=True, parents=True)
                logging.info(f"Plot extrapolation at {dir2extrapolation_plot}")
                for metric in best_metric_names:
                    if metrics[metric] < best_metrics[metric]:
                        best_metrics[metric] = metrics[metric]
                        best_models_id[metric] = run_id
                        best_models_predictions[metric] = test_predictions

                        loss = "l2"
                        observed_outcomes_only = False
                        estimated_ps = None
                        # # delete all png from the directory and save only the best
                        # for f in list(dir2extrapolation_plot.glob("*.png")):
                        #     if f.name.find("metric_"+metric+"__") != -1:
                        #         f.unlink()
                        # fig = plt.figure()
                        # ax, [ax_h_0, ax_h_1] = extrapolation_plot_residuals(
                        #     causal_df=df_test.df,
                        #     hat_mu_0=test_predictions["hat_mu_0"],
                        #     hat_mu_1=test_predictions["hat_mu_1"],
                        #     fig=fig,
                        #     loss=loss,
                        #     observed_outcomes_only=observed_outcomes_only,
                        #     estimated_ps=estimated_ps
                        # )
                        # show_estimates(
                        #     ax,
                        #     metrics,
                        #     tau=False,
                        #     tau_risk=True,
                        #     oracle_r_risk=True,
                        #     oracle_r_risk_IS2=True,
                        #     oracle_mu_iptw_risk=True,
                        #     mu_risk=True,
                        # )
                        # plt.ylim((
                        #     np.min((df_test.df["y_0"], df_test.df["y_1"])),
                        #     np.max((df_test.df["y_0"], df_test.df["y_1"]))
                        # ))
                        parameter_grid_str = (
                            candidate_model_params["meta_learner_name"]
                            + "_"
                            + dic2str(parameter_grid)
                        )
                        best_metric_params[metric] = parameter_grid_str
                        hat_ps_str = "F" if estimated_ps is None else "T"
                        # plt.savefig(dir2extrapolation_plot / f"extrapolation_plot__metric_{metric}__{parameter_grid_str}__obs_y{observed_outcomes_only}__{loss}_hat_ps={hat_ps_str}.png", bbox_inches="tight")
                        # plt.close()

    # outcome pl
    if extrapolation_plot:
        for metric_ref, metric_compared in metrics_to_compared:
            fig = plt.figure()
            ax, ax_hist = plot_metric_comparison(
                df=df_test.df,
                metric_best_predictions=best_models_predictions,
                metric_ref=metric_ref,
                metric_compared=metric_compared,
                fig=fig,
            )
            plt.savefig(
                dir2extrapolation_plot
                / f"bmodel_{metric_ref}VS{metric_compared}_pref_{best_metric_params[metric_ref]}__pcomp_{best_metric_params[metric_compared]}.png",
                bbox_inches="tight",
            )
            plt.close()

    expe_results_df = pd.concat(expe_results)

    if compute_rankings:
        dir2extrapolation_plot.mkdir(exist_ok=True, parents=True)
        xp_causal_metrics = [m for m in CAUSAL_METRICS if m in expe_results_df.columns]
        # TODO: should get this info from the dataset_name and a config file
        expe_indices = ["test_seed", "test_d_normalized_tv"]
        candidate_params = list(candidate_estimators_grid[0]["hp_kwargs"].keys()) + [
            "meta_learner_name"
        ]
        rankings = get_metric_rankings_by_dataset(
            expe_results=expe_results_df,
            expe_indices=expe_indices,
            causal_metrics=xp_causal_metrics,
            candidate_params=candidate_params,
        )
        rankings_aggregate = get_rankings_aggregate(
            expe_rankings=rankings,
            expe_indices=expe_indices,
            causal_metrics=xp_causal_metrics,
        )
        kendalltau_of_interest = {
            k.replace("kendalltau_stats__tau_risk_", ""): v
            for (k, v) in rankings_aggregate.to_dict(orient="index")[0].items()
            if k.replace("kendalltau_stats__tau_risk_", "")
            in ["oracle_r_risk_IS2", "oracle_r_risk", "oracle_mu_iptw_risk", "mu_risk"]
        }

        kendall_tau_strings = "Kendalls:\n" + "\n".join(
            [
                f"{CAUSAL_METRIC_LABELS[k]}: {v:.4f}"
                for (k, v) in kendalltau_of_interest.items()
            ]
        )
        fig = plt.figure()
        m_ref = "tau_risk"
        m_comp = "mu_risk"
        ax, ax_hist = plot_metric_comparison(
            df=df_test.df,
            metric_best_predictions=best_models_predictions,
            metric_ref=m_ref,
            metric_compared=m_comp,
            fig=fig,
            plot_annotation=False,
            show_ntv=show_ntv,
        )
        ylims = (
            np.min(np.hstack([df_test.df["mu_0"].values, df_test.df["mu_1"].values])),
            np.max(np.hstack([df_test.df["mu_0"].values, df_test.df["mu_1"].values])),
        )
        ax.text(
            s=kendall_tau_strings,
            x=0.92,
            y=0.4,
            transform=plt.gcf().transFigure,
            fontsize=27,
        )
        ax.set_ylim(ylims)
        figure_name = f"kendall_{m_ref}VS{m_comp}_pref_{best_metric_params[m_ref]}_pcomp_{best_metric_params[m_comp]}"
        plt.savefig(dir2extrapolation_plot / f"figure_name.png", bbox_inches="tight")
        save_figure_to_folders(Path("heterogeneity") / xp_name, paper_dir=True)
        plt.close()

    if extrapolation_plot:
        fig = plt.figure()
        extrapolation_plot_outcome(
            causal_df=df_test.df, observed_outcomes_only=observed_outcomes_only, fig=fig
        )
        plt.savefig(
            dir2extrapolation_plot
            / f"extrapolation_plot__outcome__obs_y{observed_outcomes_only}__{loss}_hat_ps={hat_ps_str}.png",
            bbox_inches="tight",
        )
        plt.close()
        dgp_description = df_test.describe(prefix="test_")
        dgp_description.pop("test_snr_0", None)
        dgp_description.pop("test_snr_1", None)
        for metric in best_metric_names:
            dgp_description[f"best_model_{metric}"] = best_models_id[metric]
            dgp_description[f"best_value_{metric}"] = best_metrics[metric]
        if best_models_id["tau_risk"] != best_models_id["oracle_r_risk"]:
            dgp_description["agreement__tau_risk__r_risk"] = False
        else:
            dgp_description["agreement__tau_risk__oracle_r_risk"] = True
        if best_models_id["oracle_r_risk"] != best_models_id["oracle_r_risk_IS2"]:
            dgp_description["agreement__oracle_r_risk__oracle_r_risk_IS2"] = False
        else:
            dgp_description["agreement__oracle_r_risk__oracle_r_risk_IS2"] = True
        with open(dir2extrapolation_plot / "dataset_description.json", "w") as f:
            json.dump(dgp_description, f)

    total_compute_time = datetime.now() - expe_timestamp
    end_signal = f"""
        👌 Experiment ended in {total_compute_time}, 
        -----EXPERIMENT ENDS-----\n
        """
    logger.info(end_signal)
    return expe_results_df

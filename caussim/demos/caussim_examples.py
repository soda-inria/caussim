# %%
from copy import deepcopy
from pathlib import Path
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from caussim.config import COLOR_MAPPING, DIR2PAPER_IMG, LABEL_MAPPING
from caussim.data.loading import load_dataset
from caussim.experiences.base_config import (
    CATE_CONFIG_LOGISTIC_NUISANCE,
    DEFAULT_SIMU_CONFIG,
)
from caussim.pdistances.mmd import mmd_rbf, normalized_total_variation
from caussim.reports.utils import save_figure_to_folders
from caussim.demos.utils import get_cut, plot_simu1D_cut, plot_simu2D
from caussim.data.simulations import get_transformed_space_from_simu

config_for_papers = [(187, 1, False), (8, 0.7, True)]
n_components = 2


for random_seed, overlap, legend in config_for_papers:
    sns.set(font_scale=1.7, style="whitegrid", context="talk")
    # Test for one simulation
    simu_config = deepcopy(DEFAULT_SIMU_CONFIG)
    cate_config = deepcopy(CATE_CONFIG_LOGISTIC_NUISANCE)

    # overlap = 0.7
    # random_seed = 8
    # legend=True
    simu_config["random_seed"] = random_seed
    simu_config["baseline_link"]["params"]["n_components"] = n_components
    simu_config["treatment_link"]["params"]["n_components"] = n_components
    simu_config["treatment_assignment"]["params"]["overlap"] = overlap
    # Changing initial sample seed, changes randomly the basis because we use nystroem, hence random selection in the first sampled data

    sim, df_train = load_dataset(dataset_config=simu_config)
    # Z_control, Z_treated = get_transformed_space_from_simu(sim, df_train)
    # transformed_mmd = mmd_rbf(Z_control, Z_treated)
    nTV = normalized_total_variation(df_train.df["e"], df_train.get_a().mean())
    basis = sim.baseline_pipeline.pipeline.named_steps.featurization.components_
    cut_basis_line, _ = get_cut(df_train.df, sim, basis)
    barycentre = basis.mean(axis=0)
    orthogonal_dir = np.array([basis[0, 1] - basis[1, 1], -basis[0, 0] + basis[1, 0]])
    basis_orthog = np.vstack([barycentre, barycentre + orthogonal_dir])
    cut_ortho_line, _ = get_cut(df_train.df, sim, basis_orthog)

    fig = plt.figure(constrained_layout=True, figsize=(12, 12))
    subfigs = fig.subfigures(nrows=2, ncols=1)

    subfigs[0].suptitle(
        r"Simulation: $D={}$, $\theta={}$, seed={}".format(
            n_components, overlap, random_seed
        )
    )
    ax0 = subfigs[0].subplots(nrows=1, ncols=1)
    fig, ax0 = plot_simu2D(df_train.df, sim, fig=fig, ax=ax0, legend=False, title=False)
    ax0.plot(cut_basis_line[:, 0], cut_basis_line[:, 1], c="black", lw=2)
    ax0.plot(cut_ortho_line[:, 0], cut_ortho_line[:, 1], c="black", lw=2)
    ax0.set_xlim(df_train.df["x_0"].min(), df_train.df["x_0"].max())
    ax0.set_ylim(df_train.df["x_1"].min(), df_train.df["x_1"].max())

    subfigs[1].suptitle("One-dimensional cuts of the response surfaces")
    ax10, ax11 = subfigs[1].subplots(nrows=1, ncols=2)
    fig, _ = plot_simu1D_cut(df_train.df, sim, cut_points=basis, fig=fig, ax=ax10)
    fig, _ = plot_simu1D_cut(
        df_train.df, sim, cut_points=basis_orthog, fig=fig, ax=ax11
    )
    for ax in [ax0, ax10, ax11]:
        ax.set_xticks([])
        ax.set_yticks([])
    # set custom legend
    handles = [
        Line2D(
            [0],
            [0],
            color=COLOR_MAPPING[0],
            marker="o",
            markersize=24,
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_MAPPING[1],
            marker="o",
            markersize=24,
            linestyle="None",
        ),
    ]
    labels = [LABEL_MAPPING[0], LABEL_MAPPING[1]]
    if legend:
        subfigs[0].legend(
            handles=handles,
            labels=labels,
            title="Treatment\n status",
            bbox_to_anchor=(1.3, 0.95),
            fontsize=32,
        )
    # plt.tight_layout()

    save_figure_to_folders(
        figure_name=Path(
            f"caussim_example_rs_gaussian={sim.rs_gaussian}_rs_rotation={sim.rs_rotation}_ntv={np.round(nTV, 2)}_D={n_components}_overlap={overlap}_p_A={sim.treatment_ratio}"
        ),
        paper_dir=True,
        figure_dir=True,
    )

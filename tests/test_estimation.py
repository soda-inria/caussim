from matplotlib import pyplot as plt
import numpy as np
from caussim.config import DIR2FIGURES
import pandas as pd
import pytest 

from caussim.estimation.estimation import get_selection_metrics, get_mean_outcomes_by_ps_bin, extrapolation_plot
from caussim.reports.plots_utils import CAUSAL_METRICS



def make_toy_predictions(df):
    N = df.shape[0]
    hat_mu_0 = np.random.rand(N)
    hat_mu_1 = np.random.rand(N)
    hat_m = np.random.rand(N)
    hat_e = np.random.rand(N)
    predictions = pd.DataFrame(
        {"hat_mu_1": hat_mu_1, "hat_mu_0": hat_mu_0, "check_e": hat_e, "check_m": hat_m}
    )
    predictions["hat_tau"] = hat_mu_1 - hat_mu_0
    return predictions


def test_get_selection_metrics(dummy_causal_dataset):
    predictions = make_toy_predictions(dummy_causal_dataset.df)
    selection_metrics = get_selection_metrics(dummy_causal_dataset, predictions)

    assert set(CAUSAL_METRICS).intersection(selection_metrics.keys()) == set(
        CAUSAL_METRICS
    )

def test_extrapolation_plot():
    extrapolation_plot(
        y_0=np.array([2, 2, 1, 3, 2, 1]), 
        y_1=np.array([0, 0, 1, 4, 8, 9]), 
        ps=np.array([0.1, 0.3, 0.4, 0.6, 0.80, 0.9]), 
        a=np.array([0, 0, 1, 1, 0, 1]),
        n_bins=3,
    )
    plt.savefig(DIR2FIGURES / "test_plot_extrapolation_difficulty.png", bbox_inches='tight')
    


@pytest.mark.parametrize("y_0, y_1, ps, n_bins, expected_means_outcomes, expected_sem_outcomes, expected_counts, expected_means_ps", [
    (
        np.array([2, 2, 1, 3, 2, 1]), 
        np.array([0, 0, 1, 4, 8, 9]), 
        np.array([0.1, 0.3, 0.4, 0.6, 0.80, 0.9]), 
        3,
        np.array([[2.0, 0], [2.0, 2.5], [1.5, 8.5]]), 
        np.array([[0., 0.], [1, 1.5], [0.5, 0.5]]),
        np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]), 
        np.array([[0.2, 0.2], [0.5, 0.5], [0.85, 0.85]])
    ), 
    (
        np.array([0, 0, np.NaN, np.NaN, 8, np.NaN]), 
        np.array([np.NaN, np.NaN, 1, 3, np.NaN, 1]),
        np.array([0.1, 0.3, 0.4, 0.6, 0.80, 0.9]), 
        3,
        np.array([[0, np.NaN], [np.NaN, 2.0], [8.0, 1.0]]), 
        np.array([[0., 0], [0, 1], [0, 0]]),
        np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]), 
        np.array([[0.2, np.NaN], [np.NaN, 0.5], [0.8, 0.9]])
    )
])
def test_get_mean_outcomes_by_ps_bin(y_0, y_1, ps, n_bins, expected_means_outcomes, expected_sem_outcomes, expected_counts, expected_means_ps):
    bin_means_outcomes, bin_sem_outcomes, bin_counts, bin_means_ps = get_mean_outcomes_by_ps_bin(y_0, y_1, ps, n_bins=n_bins)
    assert np.array_equal(bin_means_outcomes, expected_means_outcomes, equal_nan=True)
    assert np.array_equal(np.round(bin_sem_outcomes, 2), np.round(expected_sem_outcomes, 2), equal_nan=True)
    assert np.array_equal(bin_counts, expected_counts, equal_nan=True)
    assert np.array_equal(np.round(bin_means_ps, 2), np.round(expected_means_ps, 2), equal_nan=True)


# %%
import numpy as np

from caussim.estimation.scores import heterogeneity_score

def test_heterogeneity_score():
    h = heterogeneity_score(
        y_0=np.array([2, 2, 1, 3, 5, 2, 1]), 
        y_1=np.array([0, 0, 1, 4, 5, 8, 9]), 
        ps=np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.80, 0.9]), 
        n_bins=3,
    )
    assert np.round(h, 4) == 0.7143
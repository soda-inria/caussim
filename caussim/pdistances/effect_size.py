import numpy as np


def mean_causal_effect(mu_1: np.array, mu_0: np.array) -> float:
    """
    Compute the absolute mean standardized causal effect between two potential outcomes.
    
    This aims at measuring how much the treatment changes the outcome on a population sample.

    .. math:: \Delta_{\mu} = \frac{1}{N} \sum_{i=1}^N | \frac{\mu_{1}(x_i) - \mu_{0}(x_i)}{\mu_{0}(x_i)} |

    Args:
        mu_1 (np.array): _description_
        mu_0 (np.array): _description_

    Returns:
        float: _description_
    """
    return np.abs(mu_1/mu_0 - 1).mean()


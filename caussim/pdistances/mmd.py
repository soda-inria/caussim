import numpy as np
from sklearn.metrics import pairwise
from sklearn.utils.validation import check_array

"""
Maxmimum mean Discreptancy (MMD) with RBF kernel implementing median heuristic for bandwith (taken from [Eugenium](https://github.com/eugenium/MMD/blob/master/mmd.py)) 
"""


def mmd_rbf(X, Y):
    """
    MMD with RBF (gaussian) kernel and automatic choice of the bandwith sigma with the median heuristic. See Gretton 2012

    k(x,y) = exp(- ||x-y||^2 / (2*sigma^2))

    Args:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Returns:
        [scalar] -- [MMD value]
    """
    gamma = kernelwidth_pair(X, Y)
    return mmd_rbf_sigma_(X, Y, gamma)


def kernelwidth_pair(X, Y):
    """Implementation of the median heuristic. See Gretton 2012
    Pick sigma such that the exponent of exp(- ||x-y|| / (2*sigma2)),
    in other words ||x-y|| / (2*sigma2),  equals 1 for the median distance x
    and y of all distances between points from both data sets X and Y.
    """
    n, nfeatures = X.shape
    m, mfeatures = Y.shape

    k1 = np.sum((X * X), 1)
    q = np.tile(k1, (m, 1)).transpose()
    del k1

    k2 = np.sum((Y * Y), 1)
    r = np.tile(k2, (n, 1))
    del k2

    h = q + r
    del q, r

    # The norm
    h = h - 2 * np.dot(X, Y.transpose())
    h = np.array(h, dtype=float)

    mdist = np.median([i for i in h.flat if i])

    sigma = np.sqrt(mdist / 2.0)
    if not sigma:
        sigma = 1

    return sigma


def mmd_rbf_sigma_(X, Y, sigma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    gamma = 1 / (sigma ** 2)
    XX = pairwise.rbf_kernel(X, X, gamma)
    YY = pairwise.rbf_kernel(Y, Y, gamma)
    XY = pairwise.rbf_kernel(X, Y, gamma)
    return float(XX.mean() + YY.mean() - 2 * XY.mean())


def total_variation_distance(p, q):
    """Total variation distance between two probability distributions.
 .. math:: TV(p, q) = \frac{0.5}{N} \sum_{i=1}^N |p_i - q_i|
    Parameters
    ----------
    p : _type_
        _description_
    q : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    p = check_array(p, ensure_2d=False, dtype="numeric")
    q = check_array(q, ensure_2d=False, dtype="numeric")
    return float(0.5 * np.mean(np.abs(p - q)))


def normalized_total_variation(
    propensity_scores: np.array, treatment_probability: float
):
    """Compute a renormalized total variation distance for a population.
    
    .. math:: nTV(p(A=1|X=x), P(A=1)) = \frac{0.5}{N}\sum_{i=1}^N |\frac{p(A=1|X=x_i)}{P(A=1)} - \frac{1 - p(A=1|X=x_i)}{1 - P(A=1)} |
    
    Parameters
    ----------
    propensity_scores : np.array
        _description_
    treatment_probability : float
        _description_

    Returns
    -------
    _type_
        _description_
    """
    normalized_tv = total_variation_distance(
        propensity_scores / treatment_probability,
        (1 - propensity_scores) / (1 - treatment_probability),
    )
    return normalized_tv

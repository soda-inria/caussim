import numpy as np
from sklearn.utils.validation import check_array


def kl(p, q):
    """Naive implementation of the KL divergence.

    Args:
        p ([type]): [description]
        q ([type]): [description]

    Returns:
        [type]: [description]
    """
    p = check_array(p, ensure_2d=False, dtype="numeric")
    q = check_array(q, ensure_2d=False, dtype="numeric")
    return np.mean(np.where(p != 0, p * np.log(p / q), 0))


def jensen_shannon_divergence(p, q):
    """
    Naive implementation of the [Jensen-Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) when we have two distributions p and q over a countable set X of events.

    $$\operatorname{JSD}(P \| Q)=\frac{1}{2} D(P \| M)+\frac{1}{2} D(Q \| M)$$
    where $M=\frac{1}{2}(P+Q)$


    Args:
        p ([type]): [description]
        q ([type]): [description]

    Returns:
        [type]: [description]
    """
    M = 0.5 * (p + q)
    return 0.5 * kl(p, M) + 0.5 * kl(q, M)

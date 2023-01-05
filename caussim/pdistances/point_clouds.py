import numpy as np


def chamfer_distance(A, B):
    """Compute the symetric min distance between two point clouds, also called

    $$\mathrm{CD}\left(S_{1}, S_{2}\right)=\frac{1}{\left|S_{1}\right|} \sum_{x \in S_{1}} \min _{y \in S_{2}}\|x-y\|_{2}^{2}+\frac{1}{\left|S_{2}\right|} \sum_{y \in S_{2}} \min _{x \in S_{1}}\|x-y\|_{2}^{2}$$
    Args:
        A (_type_): _description_
        B (_type_): _description_

    Returns:
        _type_: _description_
    """
    min_distance_A_to_B = []
    for x in A:
        min_distance_A_to_B.append(np.min(np.linalg.norm(x - B, axis=1)) ** 2)
    min_distance_B_to_A = []
    for x in B:
        min_distance_B_to_A.append(np.min(np.linalg.norm(x - A, axis=1)) ** 2)
    return np.mean(min_distance_A_to_B) + np.mean(min_distance_B_to_A)

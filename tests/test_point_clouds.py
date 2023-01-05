from caussim.pdistances.point_clouds import chamfer_distance
import numpy as np


def test_chamfer_distance():
    A = np.array([[0, 5], [0, 3]])
    B = np.array([[-2, 8], [1, 4]])
    assert chamfer_distance(A, B) == 9.5

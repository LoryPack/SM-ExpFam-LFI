import unittest

import numpy as np
import torch
from scipy.stats import special_ortho_group

from src.mcc import compute_mcc


class test_default_NN_forward_derivatives(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        torch.random.manual_seed(1)
        self.x = np.random.randn(25, 5)
        self.y = np.random.uniform(size=(25, 5))
        self.x_torch = torch.randn(25, 5)
        self.y_torch = torch.rand(size=(25, 5))

    def test_mcc_works(self):
        compute_mcc(self.x, self.y, cca_dim=5, weak=True)
        compute_mcc(self.x_torch, self.y_torch, cca_dim=5, weak=True)

    def test_mcc_identical(self):
        res = compute_mcc(self.x, self.x, weak=True)
        res_torch = compute_mcc(self.x, self.x, weak=True)
        for i in range(4):
            assert np.allclose(res[i], 1)
            assert np.allclose(res_torch[i], 1)

    def test_rotation_weak(self):
        x_rotated = np.dot(self.x, special_ortho_group.rvs(5))
        # check if the weak mcc returns 1 when using a rotation matrix.
        res = compute_mcc(self.x, x_rotated, cca_dim=5, weak=True)
        # also, if I rotate x the weak mcc should not change.
        res_1 = compute_mcc(self.y, x_rotated, cca_dim=5, weak=True)
        res_2 = compute_mcc(self.y, self.x, cca_dim=5, weak=True)
        assert np.allclose(res[2], 1)
        assert np.allclose(res[3], 1)
        assert np.allclose(res_1[2], res_2[2])
        assert np.allclose(res_1[3], res_2[3], atol=1e-3)  # this may be slightly different

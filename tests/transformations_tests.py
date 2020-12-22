import unittest

import numpy as np

from src.Transformations import LowerBoundedVarScaler, TwoSidedBoundedVarScaler, BoundedVarScaler


class test_LowerBoundedVarScaler(unittest.TestCase):

    def setUp(self):
        self.seed = 10
        self.scaler = LowerBoundedVarScaler(rescale_transformed_vars=False)
        self.rng = np.random.RandomState(self.seed)
        self.x = self.rng.exponential(size=(1, 10))

    def test_scale(self):
        self.scaler.fit_transform(self.x)

    def test_inverse_transform(self):
        y = self.scaler.fit_transform(self.x)
        x2 = self.scaler.inverse_transform(y)
        self.assertTrue(np.allclose(self.x, x2))

    def test_jacobian(self):
        y = self.scaler.fit_transform(self.x)
        jac_1 = self.scaler.jac_log_det(self.x)
        jac_2 = self.scaler.jac_log_det_inverse_transform(y)
        self.assertAlmostEqual(jac_1, jac_2)


class test_TwoSidedBoundedVarScaler(unittest.TestCase):

    def setUp(self):
        self.seed = 10
        self.scaler = TwoSidedBoundedVarScaler(rescale_transformed_vars=False, upper_bound=2)
        self.rng = np.random.RandomState(self.seed)
        self.x = self.rng.uniform(size=(1, 10), high=2)

    def test_scale(self):
        self.scaler.fit_transform(self.x)

    def test_inverse_transform(self):
        y = self.scaler.fit_transform(self.x)
        x2 = self.scaler.inverse_transform(y)
        self.assertTrue(np.allclose(self.x, x2))

    def test_jacobian(self):
        y = self.scaler.fit_transform(self.x)
        jac_1 = self.scaler.jac_log_det(self.x)
        jac_2 = self.scaler.jac_log_det_inverse_transform(y)
        self.assertAlmostEqual(jac_1, jac_2)

    def test_inverse_jac_large_numbers(self):
        # test that the implementation works for large numerical values
        self.scaler.jac_log_det_inverse_transform(np.array([1000]))


class test_BoundedVarScaler(unittest.TestCase):

    def setUp(self):
        self.seed = 10
        self.two_sided_scaler = TwoSidedBoundedVarScaler(rescale_transformed_vars=False, upper_bound=2)
        self.lower_bounded_scaler = LowerBoundedVarScaler(rescale_transformed_vars=False)
        self.rng = np.random.RandomState(self.seed)
        self.x_two_sided = self.rng.uniform(size=(1, 10), high=2)
        self.x_lower_bounded = self.rng.exponential(size=(1, 10))
        self.x_unbounded = self.rng.normal(size=(1, 10))
        self.x_mixed = np.concatenate(
            (self.x_two_sided[:, 0:3], self.x_lower_bounded[:, 0:3], self.x_unbounded[:, 0:4]), axis=1)

    def test_identical_to_lower_bounded(self):
        self.scaler = BoundedVarScaler(lower_bound=np.zeros(10), upper_bound=np.array([None] * 10),
                                       rescale_transformed_vars=False)
        # test transform:
        y = self.scaler.fit_transform(self.x_lower_bounded)
        y2 = self.lower_bounded_scaler.fit_transform(self.x_lower_bounded)
        self.assertTrue(np.allclose(y, y2))
        # test inverse transform
        x_inv = self.scaler.inverse_transform(y)
        x_inv2 = self.lower_bounded_scaler.inverse_transform(y2)
        self.assertTrue(np.allclose(x_inv, x_inv2))
        self.assertTrue(np.allclose(self.x_lower_bounded, x_inv2))
        # test jacobian
        jac_1 = self.scaler.jac_log_det(self.x_lower_bounded)
        jac_2 = self.lower_bounded_scaler.jac_log_det(self.x_lower_bounded)
        self.assertAlmostEqual(jac_1, jac_2)
        # test inverse jacobian
        jac_inv_1 = self.scaler.jac_log_det_inverse_transform(y)
        jac_inv_2 = self.lower_bounded_scaler.jac_log_det_inverse_transform(y2)
        self.assertAlmostEqual(jac_inv_1, jac_inv_2)
        self.assertAlmostEqual(jac_1, jac_inv_2)

    def test_identical_to_two_sided(self):
        self.scaler = BoundedVarScaler(lower_bound=np.zeros(10), upper_bound=np.ones(10) * 2,
                                       rescale_transformed_vars=False)
        # test transform:
        y = self.scaler.fit_transform(self.x_two_sided)
        y2 = self.two_sided_scaler.fit_transform(self.x_two_sided)
        self.assertTrue(np.allclose(y, y2))
        # test inverse transform
        x_inv = self.scaler.inverse_transform(y)
        x_inv2 = self.two_sided_scaler.inverse_transform(y2)
        self.assertTrue(np.allclose(x_inv, x_inv2))
        self.assertTrue(np.allclose(self.x_two_sided, x_inv2))
        # test jacobian
        jac_1 = self.scaler.jac_log_det(self.x_two_sided)
        jac_2 = self.two_sided_scaler.jac_log_det(self.x_two_sided)
        self.assertAlmostEqual(jac_1, jac_2)
        # test inverse jacobian
        jac_inv_1 = self.scaler.jac_log_det_inverse_transform(y)
        jac_inv_2 = self.two_sided_scaler.jac_log_det_inverse_transform(y2)
        self.assertAlmostEqual(jac_inv_1, jac_inv_2)
        self.assertAlmostEqual(jac_1, jac_inv_2)

    def test_unbounded(self):
        self.scaler = BoundedVarScaler(lower_bound=np.array([None] * 10), upper_bound=np.array([None] * 10),
                                       rescale_transformed_vars=False)
        # test transform:
        y = self.scaler.fit_transform(self.x_unbounded)
        self.assertTrue(np.allclose(y, self.x_unbounded))
        # test inverse transform
        x_inv = self.scaler.inverse_transform(y)
        self.assertTrue(np.allclose(self.x_unbounded, x_inv))
        # test jacobian
        jac_1 = self.scaler.jac_log_det(self.x_unbounded)
        self.assertAlmostEqual(jac_1, 0)
        # test inverse jacobian
        jac_inv_1 = self.scaler.jac_log_det_inverse_transform(y)
        self.assertAlmostEqual(jac_inv_1, 0)

    def test_mixed(self):
        lower_bound = np.concatenate((np.zeros(6), np.array([None] * 4)), axis=0)
        upper_bound = np.concatenate((np.ones(3) * 2, np.array([None] * 7)), axis=0)
        self.scaler = BoundedVarScaler(lower_bound=lower_bound, upper_bound=upper_bound,
                                       rescale_transformed_vars=False)
        # check here if they work fine
        # test transform:
        y = self.scaler.fit_transform(self.x_mixed)
        # test inverse transform
        x_inv = self.scaler.inverse_transform(y)
        self.assertTrue(np.allclose(self.x_mixed, x_inv))
        # test jacobian
        jac_1 = self.scaler.jac_log_det(self.x_mixed)
        # self.assertAlmostEqual(jac_1, jac_2)
        # test inverse jacobian
        jac_inv_1 = self.scaler.jac_log_det_inverse_transform(y)
        # self.assertAlmostEqual(jac_inv_1, jac_inv_2)

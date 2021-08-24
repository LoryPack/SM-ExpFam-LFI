import unittest

import numpy as np
import torch

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
        self.scaler_lower_bounded = BoundedVarScaler(lower_bound=np.array([0, 0]),
                                                     upper_bound=np.array([None, None]))
        self.scaler_two_sided = BoundedVarScaler(lower_bound=np.array([0, 0]), upper_bound=np.array([10, 10]))
        self.scaler_mixed = BoundedVarScaler(lower_bound=np.array([0, 0]), upper_bound=np.array([10, None]))
        self.scaler_dummy = BoundedVarScaler(lower_bound=np.array([None, None]),
                                             upper_bound=np.array([None, None]))
        # without minmax
        self.scaler_lower_bounded_no_minmax = BoundedVarScaler(lower_bound=np.array([0, 0]),
                                                               upper_bound=np.array([None, None]),
                                                               rescale_transformed_vars=False)
        self.scaler_two_sided_no_minmax = BoundedVarScaler(lower_bound=np.array([0, 0]), upper_bound=np.array([10, 10]),
                                                           rescale_transformed_vars=False)
        self.scaler_mixed_no_minmax = BoundedVarScaler(lower_bound=np.array([0, 0]), upper_bound=np.array([10, None]),
                                                       rescale_transformed_vars=False)
        self.scaler_dummy_no_minmax = BoundedVarScaler(lower_bound=np.array([None, None]),
                                                       upper_bound=np.array([None, None]),
                                                       rescale_transformed_vars=False)

        self.list_scalers_minmax = [self.scaler_dummy, self.scaler_mixed,
                                    self.scaler_two_sided, self.scaler_lower_bounded]
        self.list_scalers_no_minmax = [self.scaler_dummy_no_minmax, self.scaler_mixed_no_minmax,
                                       self.scaler_two_sided_no_minmax, self.scaler_lower_bounded_no_minmax]

        self.list_scalers = self.list_scalers_minmax + self.list_scalers_no_minmax

        # data
        self.x = np.array([[3.2, 4.5]])
        self.x2 = np.array([[4.2, 3.5]])

    def test(self):
        for scaler in self.list_scalers:
            scaler.fit(self.x)
            self.assertEqual(self.x.shape, scaler.inverse_transform(scaler.transform(self.x)).shape)
            self.assertTrue(np.allclose(np.array(self.x), np.array(scaler.inverse_transform(scaler.transform(self.x)))))
            self.assertAlmostEqual(scaler.jac_log_det(self.x),
                                   scaler.jac_log_det_inverse_transform(scaler.transform(self.x)), delta=1e-7)

        # test dummy scaler actually does nothing:
        self.assertTrue(np.allclose(self.x, self.scaler_dummy_no_minmax.transform(self.x)))
        self.assertTrue(np.allclose(self.x, self.scaler_dummy_no_minmax.inverse_transform(self.x)))
        self.assertEqual(0, self.scaler_dummy.jac_log_det_inverse_transform(self.x))
        self.assertEqual(0, self.scaler_dummy.jac_log_det(self.x))
        self.assertEqual(0, self.scaler_dummy_no_minmax.jac_log_det_inverse_transform(self.x))
        self.assertEqual(0, self.scaler_dummy_no_minmax.jac_log_det(self.x))

        # test that the jacobian works on 1d things as well:
        self.assertEqual(0, self.scaler_dummy.jac_log_det_inverse_transform(self.x.squeeze()))
        self.assertEqual(0, self.scaler_dummy.jac_log_det(self.x.squeeze()))
        self.assertEqual(0, self.scaler_dummy_no_minmax.jac_log_det_inverse_transform(self.x.squeeze()))
        self.assertEqual(0, self.scaler_dummy_no_minmax.jac_log_det(self.x.squeeze()))

    def test_torch(self):
        # same as test but using torch input
        x_torch = torch.from_numpy(self.x)
        for scaler in self.list_scalers:
            scaler.fit(x_torch)
            self.assertEqual(x_torch.shape, scaler.inverse_transform(scaler.transform(x_torch)).shape)
            self.assertTrue(np.allclose(self.x, np.array(scaler.inverse_transform(scaler.transform(x_torch)))))
            self.assertAlmostEqual(scaler.jac_log_det(x_torch),
                                   scaler.jac_log_det_inverse_transform(scaler.transform(x_torch)), delta=1e-7)

        # test dummy scaler actually does nothing:
        self.assertTrue(np.allclose(x_torch, self.scaler_dummy_no_minmax.transform(x_torch)))
        self.assertTrue(np.allclose(x_torch, self.scaler_dummy_no_minmax.inverse_transform(x_torch)))
        self.assertEqual(0, self.scaler_dummy.jac_log_det_inverse_transform(x_torch))
        self.assertEqual(0, self.scaler_dummy.jac_log_det(x_torch))
        self.assertEqual(0, self.scaler_dummy_no_minmax.jac_log_det_inverse_transform(x_torch))
        self.assertEqual(0, self.scaler_dummy_no_minmax.jac_log_det(x_torch))

        # test that the jacobian works on 1d things as well:
        self.assertEqual(0, self.scaler_dummy.jac_log_det_inverse_transform(x_torch.squeeze()))
        self.assertEqual(0, self.scaler_dummy.jac_log_det(x_torch.squeeze()))
        self.assertEqual(0, self.scaler_dummy_no_minmax.jac_log_det_inverse_transform(x_torch.squeeze()))
        self.assertEqual(0, self.scaler_dummy_no_minmax.jac_log_det(x_torch.squeeze()))

    def test_jacobian_difference(self):
        # the values of the jacobian log det do not take into account the linear transformation as what
        # really matters are the difference between them for two x values (in an MCMC acceptance rate).
        # Then the difference of the jacobian for the same two points in original and transformed space should be
        # the same.
        for scaler_minmax, scaler_no_minmax in zip(self.list_scalers_minmax, self.list_scalers_no_minmax):
            scaler_minmax.fit(self.x)
            scaler_no_minmax.fit(self.x)

            # the difference of the log det of jacobian between two points in the original space should be the same
            self.assertAlmostEqual(
                scaler_minmax.jac_log_det(self.x) - scaler_minmax.jac_log_det(self.x2),
                scaler_no_minmax.jac_log_det(self.x) - scaler_no_minmax.jac_log_det(self.x2),
                delta=1e-7)

            # the difference of the log det of jacobian between two points corresponding to the same two points in the
            # original space (either if the linear rescaling is applied or not) should be the same
            self.assertAlmostEqual(
                scaler_minmax.jac_log_det_inverse_transform(scaler_minmax.transform(self.x)) -
                scaler_minmax.jac_log_det_inverse_transform(scaler_minmax.transform(self.x2)),
                scaler_no_minmax.jac_log_det_inverse_transform(scaler_no_minmax.transform(self.x)) -
                scaler_no_minmax.jac_log_det_inverse_transform(scaler_no_minmax.transform(self.x2)),
                delta=1e-7)

    def test_errors(self):
        with self.assertRaises(RuntimeError):
            self.scaler_mixed.jac_log_det(np.array([[1.1, 2.2], [3.3, 4.4]]))

    def test_identical_to_lower_bounded(self):
        self.scaler = LowerBoundedVarScaler(lower_bound=np.zeros(2),
                                            rescale_transformed_vars=False)
        # test transform:
        y = self.scaler.fit_transform(self.x)
        y2 = self.scaler_lower_bounded_no_minmax.fit_transform(self.x)
        self.assertTrue(np.allclose(y, y2))
        # test inverse transform
        x_inv = self.scaler.inverse_transform(y)
        x_inv2 = self.scaler_lower_bounded_no_minmax.inverse_transform(y2)
        self.assertTrue(np.allclose(x_inv, x_inv2))
        self.assertTrue(np.allclose(self.x, x_inv2))
        # test jacobian
        jac_1 = self.scaler.jac_log_det(self.x)
        jac_2 = self.scaler_lower_bounded_no_minmax.jac_log_det(self.x)
        self.assertAlmostEqual(jac_1, jac_2)
        # test inverse jacobian
        jac_inv_1 = self.scaler.jac_log_det_inverse_transform(y)
        jac_inv_2 = self.scaler_lower_bounded_no_minmax.jac_log_det_inverse_transform(y2)
        self.assertAlmostEqual(jac_inv_1, jac_inv_2)
        self.assertAlmostEqual(jac_1, jac_inv_2)

    def test_identical_to_two_sided(self):
        self.scaler = TwoSidedBoundedVarScaler(lower_bound=np.zeros(2), upper_bound=np.ones(2) * 10,
                                               rescale_transformed_vars=False)
        # test transform:
        y = self.scaler.fit_transform(self.x)
        y2 = self.scaler_two_sided_no_minmax.fit_transform(self.x)
        self.assertTrue(np.allclose(y, y2))
        # test inverse transform
        x_inv = self.scaler.inverse_transform(y)
        x_inv2 = self.scaler_two_sided_no_minmax.inverse_transform(y2)
        self.assertTrue(np.allclose(x_inv, x_inv2))
        self.assertTrue(np.allclose(self.x, x_inv2))
        # test jacobian
        jac_1 = self.scaler.jac_log_det(self.x)
        jac_2 = self.scaler_two_sided_no_minmax.jac_log_det(self.x)
        self.assertAlmostEqual(jac_1, jac_2)
        # test inverse jacobian
        jac_inv_1 = self.scaler.jac_log_det_inverse_transform(y)
        jac_inv_2 = self.scaler_two_sided_no_minmax.jac_log_det_inverse_transform(y2)
        self.assertAlmostEqual(jac_inv_1, jac_inv_2, delta=1e-6)
        self.assertAlmostEqual(jac_1, jac_inv_2, delta=1e-6)

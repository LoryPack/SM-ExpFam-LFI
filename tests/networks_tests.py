import unittest
from time import time

import torch

from src.functions import jacobian_second_order, jacobian_hessian, jacobian
from src.networks import createDefaultNNWithDerivatives, PartiallyExchangeableNetwork, createDefaultNN


class test_default_NN_forward_derivatives(unittest.TestCase):

    def setUp(self):
        self.net = createDefaultNNWithDerivatives(5, 2, nonlinearity=torch.nn.Softplus)()
        self.tensor = torch.randn((10, 5), requires_grad=True)

    def test(self):
        # compute derivative with forward pass
        y, f1, s1 = self.net.forward_and_derivatives(self.tensor)
        f2, s2 = jacobian_second_order(self.tensor, y)

        assert torch.allclose(f1, f2)
        assert torch.allclose(s1, s2)

    def test_full_hessian(self):
        # compute derivative with forward pass
        y, f1, H1 = self.net.forward_and_full_derivatives(self.tensor)
        f2, H2 = jacobian_hessian(self.tensor, y)

        assert torch.allclose(f1, f2)
        assert torch.allclose(H1, H2)


class test_PEN_forward_derivatives(unittest.TestCase):

    def setUp(self):
        self.phi_net = createDefaultNNWithDerivatives(2, 10, nonlinearity=torch.nn.Softplus,
                                                      first_derivative_only=False)()
        self.rho_net = createDefaultNNWithDerivatives(11, 2, nonlinearity=torch.nn.Softplus,
                                                      first_derivative_only=False)()

        self.net = PartiallyExchangeableNetwork(self.phi_net, self.rho_net, 1, number_timeseries=1)

        self.phi_net_order_2 = createDefaultNNWithDerivatives(3, 10, nonlinearity=torch.nn.Softplus,
                                                              first_derivative_only=False)()
        self.rho_net_order_2 = createDefaultNNWithDerivatives(12, 2, nonlinearity=torch.nn.Softplus,
                                                              first_derivative_only=False)()

        self.net_order_2 = PartiallyExchangeableNetwork(self.phi_net_order_2, self.rho_net_order_2, 2,
                                                        number_timeseries=1)

        self.phi_net_bivariate_first_der = createDefaultNNWithDerivatives(4, 10, nonlinearity=torch.nn.Softplus,
                                                                          first_derivative_only=True)()
        self.rho_net_bivariate_first_der = createDefaultNNWithDerivatives(12, 2, nonlinearity=torch.nn.Softplus,
                                                                          first_derivative_only=True)()
        self.net_bivariate_first_der = PartiallyExchangeableNetwork(self.phi_net_bivariate_first_der,
                                                                    self.rho_net_bivariate_first_der, 1,
                                                                    number_timeseries=2)
        self.phi_net_bivariate_first_der_order_2 = createDefaultNNWithDerivatives(6, 10, nonlinearity=torch.nn.Softplus,
                                                                                  first_derivative_only=True)()
        self.rho_net_bivariate_first_der_order_2 = createDefaultNNWithDerivatives(14, 2, nonlinearity=torch.nn.Softplus,
                                                                                  first_derivative_only=True)()
        self.net_bivariate_first_der_order_2 = PartiallyExchangeableNetwork(self.phi_net_bivariate_first_der_order_2,
                                                                            self.rho_net_bivariate_first_der_order_2, 2,
                                                                            number_timeseries=2)

        self.phi_net_bivariate = createDefaultNNWithDerivatives(4, 10, nonlinearity=torch.nn.Softplus,
                                                                first_derivative_only=False)()
        self.rho_net_bivariate = createDefaultNNWithDerivatives(12, 2, nonlinearity=torch.nn.Softplus,
                                                                first_derivative_only=False)()

        self.net_bivariate = PartiallyExchangeableNetwork(self.phi_net_bivariate, self.rho_net_bivariate, 1,
                                                          number_timeseries=2)

        self.phi_net_bivariate_order_2 = createDefaultNNWithDerivatives(6, 10, nonlinearity=torch.nn.Softplus,
                                                                        first_derivative_only=False)()
        self.rho_net_bivariate_order_2 = createDefaultNNWithDerivatives(14, 2, nonlinearity=torch.nn.Softplus,
                                                                        first_derivative_only=False)()

        self.net_bivariate_order_2 = PartiallyExchangeableNetwork(self.phi_net_bivariate_order_2,
                                                                  self.rho_net_bivariate_order_2, 2,
                                                                  number_timeseries=2)
        self.phi_net_trivariate_order_4 = createDefaultNNWithDerivatives(15, 10, nonlinearity=torch.nn.Softplus,
                                                                         first_derivative_only=False)()
        self.rho_net_trivariate_order_4 = createDefaultNNWithDerivatives(22, 2, nonlinearity=torch.nn.Softplus,
                                                                         first_derivative_only=False)()

        self.net_trivariate_order_4 = PartiallyExchangeableNetwork(self.phi_net_trivariate_order_4,
                                                                   self.rho_net_trivariate_order_4, 4,
                                                                   number_timeseries=3)

        self.tensor = torch.randn((10, 30), requires_grad=True)

    def test_r_block_switch_invariant(self):
        """We test here if the output of the net is r-block-switch invariant, with r=2"""

        # define two tensors that differ only for a 2-block-switch transformation
        tensor1 = torch.tensor([1, 7, 2, 3, 6, 4, 5, 8, 1, 7, 7, 2, 9, 5, 8, 1], dtype=torch.float).reshape(1, -1)
        tensor2 = torch.tensor([1, 7, 2, 9, 5, 8, 1, 7, 7, 2, 3, 6, 4, 5, 8, 1], dtype=torch.float).reshape(1, -1)
        # this is instead different
        tensor3 = torch.tensor([1, 7, 3, 9, 5, 8, 1, 7, 7, 2, 3, 6, 4, 5, 8, 1], dtype=torch.float).reshape(1, -1)

        out1 = self.net_order_2(tensor1)
        out2 = self.net_order_2(tensor2)
        out3 = self.net_order_2(tensor3)

        assert torch.allclose(out1, out2)
        assert not torch.allclose(out1, out3)

    def test_output(self):
        # check if output is kept the same
        y, f1, s1 = self.net.forward_and_derivatives(self.tensor)
        y1 = self.net(self.tensor)
        assert torch.allclose(y, y1)

    def test_output_bivariate(self):
        # check if output is kept the same
        y, f1, s1 = self.net_bivariate.forward_and_derivatives(self.tensor)
        y1 = self.net_bivariate(self.tensor)
        assert torch.allclose(y, y1)

    def test_output_bivariate_order_2(self):
        # check if output is kept the same
        y, f1, s1 = self.net_bivariate_order_2.forward_and_derivatives(self.tensor)
        y1 = self.net_bivariate_order_2(self.tensor)
        assert torch.allclose(y, y1)

    def test(self):
        # compute derivative with forward pass
        y, f1, s1 = self.net.forward_and_derivatives(self.tensor)
        f2, s2 = jacobian_second_order(self.tensor, y)

        assert torch.allclose(f1, f2)
        assert torch.allclose(s1, s2)

    def test_order_2(self):
        # compute derivative with forward pass
        y, f1, s1 = self.net_order_2.forward_and_derivatives(self.tensor)
        f2, s2 = jacobian_second_order(self.tensor, y)

        assert torch.allclose(f1, f2)
        assert torch.allclose(s1, s2)

    def test_bivariate_first_only(self):
        # compute derivative with forward pass
        y, f1 = self.net_bivariate_first_der.forward_and_first_der(self.tensor)
        f2 = jacobian(self.tensor, y)

        assert torch.allclose(f1, f2)

    def test_bivariate_first_only_order_2(self):
        # compute derivative with forward pass
        y, f1 = self.net_bivariate_first_der_order_2.forward_and_first_der(self.tensor)
        f2 = jacobian(self.tensor, y)

        assert torch.allclose(f1, f2)

    def test_bivariate(self):
        # compute derivative with forward pass
        y, f1, s1 = self.net_bivariate.forward_and_derivatives(self.tensor)
        f2, s2 = jacobian_second_order(self.tensor, y)

        assert torch.allclose(f1, f2)
        assert torch.allclose(s1, s2)

    def test_bivariate_order_2(self):
        # compute derivative with forward pass
        y, f1, s1 = self.net_bivariate_order_2.forward_and_derivatives(self.tensor)
        f2, s2 = jacobian_second_order(self.tensor, y)

        assert torch.allclose(f1, f2)
        assert torch.allclose(s1, s2)

    def test_trivariate_order_4(self):
        # compute derivative with forward pass
        y, f1, s1 = self.net_trivariate_order_4.forward_and_derivatives(self.tensor)
        f2, s2 = jacobian_second_order(self.tensor, y)

        assert torch.allclose(f1, f2)
        assert torch.allclose(s1, s2)


class test_jacobian_with_batchnorm(unittest.TestCase):
    def setUp(self) -> None:
        self.net = createDefaultNN(20, 2, nonlinearity=torch.nn.Softplus(), batch_norm_last_layer=False)()
        self.net_batchnorm = createDefaultNN(20, 2, nonlinearity=torch.nn.Softplus(), batch_norm_last_layer=True)()
        self.net_batchnorm_no_affine = createDefaultNN(20, 2, nonlinearity=torch.nn.Softplus(),
                                                       batch_norm_last_layer=True, affine_batch_norm=False)()
        self.data = torch.randn((15, 20), requires_grad=True)

    def test_equality_Hessian_jacobian_second(self):
        y = self.net(self.data)
        f_x, s_x = jacobian_second_order(self.data, y)
        f_x_1, H_x = jacobian_hessian(self.data, y)
        s_x_1 = torch.einsum('biik->bik', H_x)

        assert torch.allclose(f_x, f_x_1)
        assert torch.allclose(s_x, s_x_1)

    def test_equality_Hessian_jacobian_second_bn(self):
        y = self.net_batchnorm(self.data)
        f_x, s_x = jacobian_second_order(self.data, y)
        f_x_1, H_x = jacobian_hessian(self.data, y)
        s_x_1 = torch.einsum('biik->bik', H_x)

        assert torch.allclose(f_x, f_x_1)
        assert torch.allclose(s_x, s_x_1)

    def test_equality_Hessian_jacobian_second_bn_no_affine(self):
        y = self.net_batchnorm_no_affine(self.data)
        f_x, s_x = jacobian_second_order(self.data, y)
        f_x_1, H_x = jacobian_hessian(self.data, y)
        s_x_1 = torch.einsum('biik->bik', H_x)

        assert torch.allclose(f_x, f_x_1)
        assert torch.allclose(s_x, s_x_1)

    def test_running_time(self):
        start = time()
        for i in range(100):
            y = self.net(self.data)
            f2, s2 = jacobian_second_order(self.data, y)
        time_no_bn = time() - start
        start = time()
        for i in range(100):
            y1 = self.net_batchnorm(self.data)
            f2, s2 = jacobian_second_order(self.data, y1)
        time_bn = time() - start
        start = time()
        for i in range(100):
            y2 = self.net_batchnorm(self.data)
            f2, s2 = jacobian_second_order(self.data, y2)
        time_bn_no_affine = time() - start

        print("No bn: {:.4f}, bn: {:.4f}, bn no affine: {:.4f}".format(time_no_bn, time_bn, time_bn_no_affine))
        # computing the gradients with a batchnorm layer is much slower, and it is even slower if using the affine learned transformation.

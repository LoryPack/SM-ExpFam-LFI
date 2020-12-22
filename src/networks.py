import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


# the following is a function defining a class of fully connected NNs with given parameters; actually this could be
# merged with the next one, but we leave it separate for compatibility for now

def createDefaultNN(input_size, output_size, hidden_sizes=None, nonlinearity=None, nonlinearity_last_layer=False,
                    batch_norm=False, batch_norm_last_layer=False, affine_batch_norm=True,
                    batch_norm_last_layer_momentum=0.1, add_input_at_the_end=False):
    """Function returning a fully connected neural network class with a given input and output size, and optionally
    given hidden layer sizes (if these are not given, they are determined from the input and output size with some
    expression.

    In order to instantiate the network, you need to write: createDefaultNN(input_size, output_size)() as the function
    returns a class, and () is needed to instantiate an object.

    Note that the nonlinearity here is as an object or a functional, not a class, eg:
        nonlinearity =  nn.Softplus()
    or:
        nonlinearity =  nn.functional.softplus()
    """

    class DefaultNN(nn.Module):
        """Neural network class with sizes determined by the upper level variables."""

        def __init__(self):
            super(DefaultNN, self).__init__()
            # put some fully connected layers:

            if hidden_sizes is not None and len(hidden_sizes) == 0:
                # it is effectively a linear network
                self.fc_in = nn.Linear(input_size, output_size)

            else:
                if hidden_sizes is None:
                    # then set some default values for the hidden layers sizes; is this parametrization reasonable?
                    hidden_sizes_list = [int(input_size * 1.5), int(input_size * 0.75 + output_size * 3),
                                         int(output_size * 5)]

                else:
                    hidden_sizes_list = hidden_sizes

                self.fc_in = nn.Linear(input_size, hidden_sizes_list[0])

                # define now the hidden layers
                self.fc_hidden = nn.ModuleList()
                for i in range(len(hidden_sizes_list) - 1):
                    self.fc_hidden.append(nn.Linear(hidden_sizes_list[i], hidden_sizes_list[i + 1]))
                self.fc_out = nn.Linear(hidden_sizes_list[-1], output_size)

                # define the batch_norm:
                if batch_norm:
                    self.bn_in = nn.BatchNorm1d(hidden_sizes_list[0])
                    self.bn_hidden = nn.ModuleList()
                    for i in range(len(hidden_sizes_list) - 1):
                        self.bn_hidden.append(nn.BatchNorm1d(hidden_sizes_list[i + 1]))
                if batch_norm_last_layer:
                    self.bn_out = nn.BatchNorm1d(output_size, affine=affine_batch_norm,
                                                 momentum=batch_norm_last_layer_momentum)

        def forward(self, x):

            if add_input_at_the_end:
                x_0 = x.clone().detach()  # this may not be super efficient, but for now it is fine

            if nonlinearity is None:
                nonlinearity_fcn = F.relu
            else:
                nonlinearity_fcn = nonlinearity

            if not hasattr(self,
                           "fc_hidden"):  # it means that hidden sizes was provided and the length of the list was 0
                x = self.fc_in(x)
                if nonlinearity_last_layer:
                    x = nonlinearity_fcn(x)
                return x

            x = nonlinearity_fcn(self.fc_in(x))
            if batch_norm:
                x = self.bn_in(x)
            for i in range(len(self.fc_hidden)):
                x = nonlinearity_fcn(self.fc_hidden[i](x))
            if batch_norm:
                x = self.bn_hidden[i](x)

            x = self.fc_out(x)
            if add_input_at_the_end:
                x += x_0  # add input (before batch_norm)
            if batch_norm_last_layer:
                x = self.bn_out(x)
            if nonlinearity_last_layer:
                x = nonlinearity_fcn(x)
            return x

    return DefaultNN


# the following is a function defining a class of fully connected NNs with given parameters, which computes the first
# and second derivative of output wrt input along with the forward pass.

def createDefaultNNWithDerivatives(input_size, output_size, hidden_sizes=None, nonlinearity=None,
                                   nonlinearity_last_layer=False, first_derivative_only=False):
    """Function returning a fully connected neural network class with a given input and output size, and optionally
    given hidden layer sizes (if these are not given, they are determined from the input and output size with some
    expression. This neural network is capable of computing the first and second derivatives of output with respect to
    input along with the forward pass.

    All layers in this meural network are linear.

    In order to instantiate the network, you need to write: createDefaultNN(input_size, output_size)() as the function
    returns a class, and () is needed to instantiate an object.

    Note that the nonlinearity here is passed as a class, not an object, eg:
        nonlinearity =  nn.Softplus
    """

    # TODO here we still need to implement the computation of derivatives in the case of nonlinearity_last_layer=True.
    class DefaultNNWithDerivatives(nn.Module):
        """Neural network class with sizes determined by the upper level variables."""

        def __init__(self):
            super(DefaultNNWithDerivatives, self).__init__()
            # put some fully connected layers:

            if nonlinearity is None:  # default nonlinearity
                non_linearity = nn.ReLU
            else:
                non_linearity = nonlinearity  # need to change nome otherwise it gives Error

            if hidden_sizes is not None and len(hidden_sizes) == 0:
                # it is effectively a linear network
                self.fc_in = nn.Linear(input_size, output_size)
                if nonlinearity_last_layer:
                    self.nonlinearity_in = non_linearity()

            else:
                if hidden_sizes is None:
                    # then set some default values for the hidden layers sizes; is this parametrization reasonable?
                    hidden_sizes_list = [int(input_size * 1.5), int(input_size * 0.75 + output_size * 3),
                                         int(output_size * 5)]

                else:
                    hidden_sizes_list = hidden_sizes

                self.fc_in = nn.Linear(input_size, hidden_sizes_list[0])
                self.nonlinearity_in = non_linearity()

                # define now the hidden layers
                self.fc_hidden = nn.ModuleList()
                self.nonlinearities_hidden = nn.ModuleList()
                for i in range(len(hidden_sizes_list) - 1):
                    self.fc_hidden.append(nn.Linear(hidden_sizes_list[i], hidden_sizes_list[i + 1]))
                    self.nonlinearities_hidden.append(non_linearity())
                self.fc_out = nn.Linear(hidden_sizes_list[-1], output_size)
                if nonlinearity_last_layer:
                    self.nonlinearity_out = non_linearity()

        def forward(self, x):

            if not hasattr(self,
                           "fc_hidden"):  # it means that hidden sizes was provided and the length of the list was 0, ie the
                if nonlinearity_last_layer:
                    x = self.fc_in(x)
                    x1 = self.nonlinearity_in(x)
                    return x1
                else:
                    return self.fc_in(x)

            x = self.fc_in(x)
            x1 = self.nonlinearity_in(x)

            for i in range(len(self.fc_hidden)):
                x = self.fc_hidden[i](x1)
                x1 = self.nonlinearities_hidden[i](x)

            x = self.fc_out(x1)
            if nonlinearity_last_layer:
                x = nonlinearity(x)

            return x

        def forward_and_derivatives(self, x):

            # initialize the derivatives:
            f = self.fc_in.weight.unsqueeze(0).repeat(x.shape[0], 1, 1).transpose(2, 1).transpose(0,
                                                                                                  1)  # one for each element of the batch
            if not first_derivative_only:
                s = torch.zeros_like(f)

            if not hasattr(self, "fc_hidden"):
                # it means that hidden sizes was provided and the length of the list was 0, ie the net is a single layer.
                if nonlinearity_last_layer:
                    raise NotImplementedError
                    x = self.fc_in(x)
                    x1 = self.nonlinearity_in(x)
                    return x1  # ????
                else:
                    if first_derivative_only:
                        return self.fc_in(x), f.transpose(0, 1)
                    else:
                        return self.fc_in(x), f.transpose(0, 1), s.transpose(0, 1)

            x = self.fc_in(x)
            x1 = self.nonlinearity_in(x)

            # TODO this implementation does not work for softsign and tanhshrink non linearity.
            for i in range(len(self.fc_hidden)):
                z = x1.grad_fn(torch.ones_like(x1))  # here we repeat some computation from the above line
                # z = grad(x1, x, torch.ones_like(x1), create_graph=True)[0]  # here we repeat some computation from the above line
                # you need to update first the second derivative, as you need the first derivative at previous layer
                if not first_derivative_only:
                    s = z * s + grad(z, x, torch.ones_like(z), retain_graph=True)[0] * f ** 2
                f = z * f
                f = F.linear(f, self.fc_hidden[i].weight)
                if not first_derivative_only:
                    s = F.linear(s, self.fc_hidden[i].weight)

                x = self.fc_hidden[i](x1)
                x1 = self.nonlinearities_hidden[i](x)

            z = x1.grad_fn(torch.ones_like(x1))  # here we repeat some computation from the above line
            # z = grad(x1, x, torch.ones_like(x1), create_graph=True)[0]  # here we repeat some computation from the above line
            # you need to update first the second derivative, as you need the first derivative at previous layer
            if not first_derivative_only:
                s = z * s + grad(z, x, torch.ones_like(z), retain_graph=True)[0] * f ** 2
            f = z * f
            f = F.linear(f, self.fc_out.weight)
            if not first_derivative_only:
                s = F.linear(s, self.fc_out.weight)

            x = self.fc_out(x1)
            if nonlinearity_last_layer:
                raise NotImplementedError
                x = nonlinearity(x)
                # need to change something here!!!

            if first_derivative_only:
                return x, f.transpose(0, 1)
            else:
                return x, f.transpose(0, 1), s.transpose(0, 1)

        def forward_and_full_derivatives(self, x):
            """This computes jacobian and full Hessian matrix"""

            # initialize the derivatives (one for each element of the batch)
            f = self.fc_in.weight.unsqueeze(0).repeat(x.shape[0], 1, 1).transpose(2, 1).transpose(0, 1)
            H = torch.zeros((f.shape[0], *f.shape)).to(f)  # hessian has an additional dimension wrt f

            if not hasattr(self, "fc_hidden"):
                # it means that hidden sizes was provided and the length of the list was 0, ie the net is a single layer
                if nonlinearity_last_layer:
                    raise NotImplementedError
                    x = self.fc_in(x)
                    x1 = self.nonlinearity_in(x)
                    return x1  # ????
                else:
                    return self.fc_in(x), f.transpose(0, 1), H.transpose(0, 2)

            x = self.fc_in(x)
            x1 = self.nonlinearity_in(x)

            # TODO this implementation does not work for softsign and tanhshrink non linearity.
            for i in range(len(self.fc_hidden)):
                z = x1.grad_fn(torch.ones_like(x1))  # here we repeat some computation from the above line
                # print("H", H.shape, "z", z.shape, "z'", grad(z, x, torch.ones_like(z), retain_graph=True)[0].shape, "f", f.shape)
                # z = grad(x1, x, torch.ones_like(x1), create_graph=True)[0]  # here we repeat some computation from the above line
                # you need to update first the second derivative, as you need the first derivative at previous layer
                H = z * H + grad(z, x, torch.ones_like(z), retain_graph=True)[0] * torch.einsum('ibo,jbo->ijbo', f, f)
                f = z * f
                f = F.linear(f, self.fc_hidden[i].weight)
                H = F.linear(H, self.fc_hidden[i].weight)

                x = self.fc_hidden[i](x1)
                x1 = self.nonlinearities_hidden[i](x)

            z = x1.grad_fn(torch.ones_like(x1))  # here we repeat some computation from the above line
            # z = grad(x1, x, torch.ones_like(x1), create_graph=True)[0]  # here we repeat some computation from the above line
            # you need to update first the second derivative, as you need the first derivative at previous layer
            H = z * H + grad(z, x, torch.ones_like(z), retain_graph=True)[0] * torch.einsum('ibo,jbo->ijbo', f, f)
            f = z * f
            f = F.linear(f, self.fc_out.weight)
            H = F.linear(H, self.fc_out.weight)
            x = self.fc_out(x1)
            if nonlinearity_last_layer:
                raise NotImplementedError
                x = nonlinearity(x)
                # need to change something here!!!

            return x, f.transpose(0, 1), H.transpose(0, 2)

    return DefaultNNWithDerivatives


class PartiallyExchangeableNetwork(nn.Module):
    """We implement this for a multivariate timeseries; this assumes that the original timeseries had been flattened
    (so it will reshape to the correct shape).
    The only constraints on phi_net, rho_net are that:

    - phi_net has `(order + 1) * number_timeseries' input neurons (ie if your univariate time series is 2-Markovian, need 3)

    - rho_net has `order * number_timeseries + output of phi_net` input neurons

    - rho_net has output neurons corresponding to the number of parameters to be estimated"""

    # Note: this has been tested for a multivariate timeseries of order 1 and univariate of order > 1, but never for
    # multivariate of order > 1.
    def __init__(self, phi_net, rho_net, order, number_timeseries=1):
        super(PartiallyExchangeableNetwork, self).__init__()
        self.phi_net = phi_net
        self.rho_net = rho_net
        self.order = order
        self.number_timeseries = number_timeseries

    def forward(self, x):
        """x should be a tensor of shape (n_samples, timestep). """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            raise RuntimeError("The input must have 2 dimensions.")

        # reshape adding back the additional dimension (we assume here that the timeseries was flattened before).
        x = x.reshape(x.shape[0], self.number_timeseries, -1)

        timestep = x.shape[-1]
        # print(x.shape)

        # format the stuff so that you can apply it to the inner network.
        input_inner = torch.cat([x[:, :, i:timestep - self.order + i] for i in range(self.order + 1)], 1).transpose(2,
                                                                                                                    1)
        # print(input_inner.shape)

        # apply inner network
        output_inner = self.phi_net(input_inner)
        # print(output_inner.shape)

        # sum along timesteps
        output_inner = torch.sum(output_inner, dim=1)
        # print(output_inner.shape)

        # concatenate each sample to the first timestep
        input_outer = torch.cat((x[:, :, 0:self.order].reshape(x.shape[0], -1), output_inner), dim=1)
        # print(input_outer.shape)

        # apply outer network:
        output_outer = self.rho_net(input_outer)
        # print(output_outer.shape)

        return output_outer

    def forward_and_first_der(self, x):
        """x should be a tensor of shape (n_samples, timestep)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            raise RuntimeError("The input must have 2 dimensions.")

        # reshape adding an additional dimension.
        x = x.reshape(x.shape[0], self.number_timeseries, -1)

        timestep = x.shape[-1]

        # format the stuff so that you can apply it to the inner network.
        input_inner = torch.cat([x[:, :, i:timestep - self.order + i] for i in range(self.order + 1)], 1).transpose(2,
                                                                                                                    1)
        input_shape = input_inner.shape

        # apply inner network (with derivatives as well); need to reshape in order to get derivatives as well!
        output_inner, f_inner = self.phi_net.forward_and_derivatives(input_inner.reshape(-1, input_shape[-1]))
        output_inner = output_inner.reshape((input_shape[0], input_shape[1], -1))
        # manipulate the derivative; add an index with number_timeseries elements:
        f_inner = f_inner.reshape((input_shape[0], input_shape[1], -1, self.number_timeseries, f_inner.shape[-1]))
        f_inner_shape = f_inner.shape
        f_inner = torch.cat((f_inner, torch.zeros(f_inner_shape[0], self.order, f_inner_shape[2], f_inner_shape[3],
                                                  f_inner_shape[4]).to(f_inner)), dim=1)
        f_inner = torch.cat(
            [torch.roll(f_inner[:, :, i, :, :].unsqueeze(2), i, dims=1) for i in range(f_inner.shape[2])],
            dim=2)
        f_inner = torch.sum(f_inner, dim=2)
        # transpose now:
        f_inner = f_inner.transpose(2, 1)

        # EXPERIMENTAL VERSION (NOT WORKING):
        # f_inner = f_inner.transpose(3,1)
        # f_inner_shape = f_inner.shape
        # print(f_inner.shape)
        # kernel = torch.flip(torch.eye(f_inner_shape[2]), (0,1))[None, None,...].repeat(f_inner_shape[1], 1, 1, 1)
        # print(kernel.shape)
        # conv_diag_sums = F.conv2d(f_inner, kernel, padding=(f_inner_shape[3]-1, f_inner_shape[2]-1), groups=f_inner_shape[1])[..., 0, :]
        # conv_diag_sums = conv_diag_sums.transpose(1,2)
        # print(conv_diag_sums.shape)
        # f_inner = torch.sum()
        # f_inner = torch.roll(f_inner, )

        # sum along timesteps
        output_inner = torch.sum(output_inner, dim=1)
        # concatenate each sample to the first self.order timesteps
        input_outer = torch.cat((x[:, :, 0:self.order].reshape(x.shape[0], -1), output_inner), dim=1)

        # apply outer network (with derivatives):
        output_outer, f_rho = self.rho_net.forward_and_derivatives(input_outer)

        f_rho_first_part = f_rho[:, 0:self.order * self.number_timeseries, :]
        f_rho_first_part = f_rho_first_part.reshape((f_rho_first_part.shape[0], self.number_timeseries, self.order, -1))
        f_rho_second_part = f_rho[:, self.order * self.number_timeseries:, :]
        f = torch.bmm(f_inner.reshape(f_inner.shape[0], -1, f_inner.shape[-1]), f_rho_second_part)

        f = f.reshape(f.shape[0], self.number_timeseries, -1, f.shape[-1])

        f[:, :, 0:self.order, :] += f_rho_first_part
        # reshape here
        return output_outer, f.reshape((f.shape[0], -1, f.shape[-1]))

    def forward_and_derivatives(self, x):
        """x should be a tensor of shape (n_samples, timestep)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            raise RuntimeError("The input must have 2 dimensions.")

        # reshape adding an additional dimension.
        x = x.reshape(x.shape[0], self.number_timeseries, -1)

        timestep = x.shape[-1]

        # format the stuff so that you can apply it to the inner network.
        input_inner = torch.cat([x[:, :, i:timestep - self.order + i] for i in range(self.order + 1)], 1).transpose(2,
                                                                                                                    1)
        input_shape = input_inner.shape

        # apply inner network (with derivatives as well); need to reshape in order to get derivatives as well (in fact
        # the network with forward derivative computation only works on 2D inputs)
        output_inner, f_inner, s_inner = self.phi_net.forward_and_derivatives(input_inner.reshape(-1, input_shape[-1]))
        output_inner = output_inner.reshape((input_shape[0], input_shape[1], -1))
        # manipulate the derivative; add an index with number_timeseries elements:
        f_inner = f_inner.reshape((input_shape[0], input_shape[1], -1, self.number_timeseries, f_inner.shape[-1]))
        f_inner_shape = f_inner.shape
        f_inner = torch.cat((f_inner, torch.zeros(f_inner_shape[0], self.order, f_inner_shape[2], f_inner_shape[3],
                                                  f_inner_shape[4]).to(f_inner)), dim=1)
        f_inner = torch.cat(
            [torch.roll(f_inner[:, :, i, :, :].unsqueeze(2), i, dims=1) for i in range(f_inner.shape[2])],
            dim=2)
        f_inner = torch.sum(f_inner, dim=2)
        # transpose now:
        f_inner = f_inner.transpose(2, 1)
        # same for second derivative:
        s_inner = s_inner.reshape((input_shape[0], input_shape[1], -1, self.number_timeseries, s_inner.shape[-1]))
        s_inner_shape = s_inner.shape
        s_inner = torch.cat((s_inner, torch.zeros(s_inner_shape[0], self.order, s_inner_shape[2], s_inner_shape[3],
                                                  s_inner_shape[4]).to(s_inner)), dim=1)
        s_inner = torch.cat(
            [torch.roll(s_inner[:, :, i, :, :].unsqueeze(2), i, dims=1) for i in range(s_inner.shape[2])],
            dim=2)
        s_inner = torch.sum(s_inner, dim=2)
        # transpose now:
        s_inner = s_inner.transpose(2, 1)

        # sum along timesteps
        output_inner = torch.sum(output_inner, dim=1)
        # concatenate each sample to the first self.order timesteps
        input_outer = torch.cat((x[:, :, 0:self.order].reshape(x.shape[0], -1), output_inner), dim=1)

        # apply outer network (with derivatives):
        output_outer, f_rho, H_rho = self.rho_net.forward_and_full_derivatives(input_outer)
        # this contains the second derivatives of outputs wrt inputs:
        s_rho = torch.diagonal(H_rho, dim1=1, dim2=2).transpose(1, 2)
        # these are for the terms which apply to the first self.order elements only
        f_rho_first_part = f_rho[:, 0:self.order * self.number_timeseries, :]
        f_rho_first_part = f_rho_first_part.reshape((f_rho_first_part.shape[0], self.number_timeseries, self.order, -1))
        f_rho_second_part = f_rho[:, self.order * self.number_timeseries:, :]
        f = torch.bmm(f_inner.reshape(f_inner.shape[0], -1, f_inner.shape[-1]), f_rho_second_part)

        f = f.reshape(f.shape[0], self.number_timeseries, -1, f.shape[-1])
        # correct with the terms which are for the first self.order terms only
        f[:, :, 0:self.order, :] += f_rho_first_part

        # need to take care to do this in the right order; these are for the terms which apply to the first self.order
        # elements only
        f_z_part = f_inner[:, :, 0:self.order, :]
        f_z_part = f_z_part.reshape(f_z_part.shape[0], -1, f_z_part.shape[-1])
        H_rho_part = H_rho[:, 0:self.order * self.number_timeseries, self.order * self.number_timeseries:, :]

        # reshape back:
        f_inner = f_inner.reshape(f_inner.shape[0], -1, f_inner.shape[-1])
        s_inner = s_inner.reshape(s_inner.shape[0], -1, s_inner.shape[-1])

        s = torch.bmm(s_inner, f_rho[:, self.order * self.number_timeseries:, :]) + \
            torch.einsum('bijk,bli,blj->blk',
                         H_rho[:, self.order * self.number_timeseries:, self.order * self.number_timeseries:, :],
                         f_inner, f_inner)

        s = s.reshape(s.shape[0], self.number_timeseries, -1, s.shape[-1])
        # correct with the terms which are for the first self.order terms only
        s_rho_first_part = s_rho[:, 0:self.order * self.number_timeseries, :]
        s_rho_first_part = s_rho_first_part.reshape((s_rho_first_part.shape[0], self.number_timeseries, self.order, -1))
        s[:, :, 0:self.order, :] += s_rho_first_part + 2 * torch.einsum('bikj,bik->bij', H_rho_part, f_z_part).reshape(
            x.shape[0], self.number_timeseries, self.order, -1)

        return output_outer, f.reshape((f.shape[0], -1, f.shape[-1])), s.reshape((f.shape[0], -1, f.shape[-1]))


def create_PEN_architecture(phi_net, rho_net, order, number_timeseries=1):
    def instantiate_PEN_net():
        return PartiallyExchangeableNetwork(phi_net(), rho_net(), order, number_timeseries=number_timeseries)

    return instantiate_PEN_net


class IdentityNet(nn.Module):
    def __init__(self):
        super(IdentityNet, self).__init__()
        # to have a parameter in the net, otherwise the general code does not work:
        self.layer1 = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return x

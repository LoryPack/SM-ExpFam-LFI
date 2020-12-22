import numpy as np
import torch
import torch.nn as nn
from abcpy.continuousmodels import Normal
from abcpy.statistics import Statistics


# generate one single gaussian sample given a theta
def generate_single_gaussian_sample(theta, rng=None):
    if rng is None:
        return np.random.normal(loc=theta[0], scale=theta[1], size=10)
    else:
        return rng.normal(loc=theta[0], scale=theta[1], size=10)


# generate Gaussian training examples:
def generate_gaussian_training_samples(n_theta=50, size_iid_samples=10, seed=None,
                                       mu_bounds=[0, 1], sigma_bounds=[0, 1]):
    """Generate parameter simulation pairs. The parameters are (mu, sigma) and have a uniform prior with bounds given
    by mu_bounds, sigma_bounds. This returns a torch tensor. sigma is the standard deviation"""

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    theta_vect = torch.cat((mu_bounds[0] + torch.rand(n_theta, 1) * (mu_bounds[1] - mu_bounds[0]),
                            sigma_bounds[0] + torch.rand(n_theta, 1) * (sigma_bounds[1] - sigma_bounds[0])), 1)

    samples_matrix = np.zeros((n_theta, size_iid_samples))  # n_parameters x size_iid_samples
    for i in range(samples_matrix.shape[0]):
        samples_matrix[i, :] = np.random.normal(loc=theta_vect[i, 0], scale=theta_vect[i, 1],
                                                size=(size_iid_samples))

    samples_matrix = torch.tensor(samples_matrix, requires_grad=False, dtype=torch.float)

    return theta_vect, samples_matrix


def compute_natural_params_Gaussian(theta_vect):
    n_theta = theta_vect.shape[0]
    return torch.cat((torch.tensor([theta_vect[i, 0] / theta_vect[i, 1] ** 2 for i in range(n_theta)]).view(-1, 1),
                      -0.5 / theta_vect[:, 1].view(-1, 1)), dim=1)


# THE FOLLOWING THINGS ARE NEEDED FOR ABCpy inference
class IidNormal(Normal):
    # this is the ABCpy model for an iid gaussian sample.
    def __init__(self, parameters, iid_size=1, name='Iid_Normal'):
        self.iid_size = iid_size
        super(IidNormal, self).__init__(parameters, name)

    def forward_simulate(self, input_values, k, rng=np.random.RandomState()):
        mu = input_values[0]
        sigma = input_values[1]
        result = np.array(rng.normal(mu, sigma, (k, self.iid_size)))
        return [np.array([x]).reshape(-1, ) for x in result]

    def get_output_dimension(self):
        return self.iid_size


def extract_params_and_weights_from_journal_gaussian(jrnl, step=None):
    params = jrnl.get_parameters(step)
    weights = jrnl.get_weights(step)
    return np.concatenate((np.array(params['mu']).reshape(-1, 1), np.array(params['sigma']).reshape(-1, 1)),
                          axis=1), weights.reshape(-1)


def extract_params_from_journal_gaussian(jrnl, step=None):
    params = jrnl.get_parameters(step)
    return np.concatenate((np.array(params['mu']).reshape(-1, 1), np.array(params['sigma']).reshape(-1, 1)), axis=1)


def extract_posterior_mean_from_journal_gaussian(jrnl, step=None):
    post_mean = jrnl.posterior_mean(step)
    return np.array((post_mean['mu'], post_mean['sigma']))


class GaussianStatistics(Statistics):
    """
    This class implements identity statistics not applying any transformation to the data, before the optional
    polynomial expansion step. If the data set contains n numpy.ndarray of length p, it returns therefore an
    nx(p+degree*p+cross*nchoosek(p,2)) matrix, where for each of the n points with p statistics, degree*p polynomial
    expansion term and cross*nchoosek(p,2) many cross-product terms are calculated.
    """

    def __init__(self, previous_statistics=None):

        self.previous_statistics = previous_statistics

    def statistics(self, data):

        # pipeline: first call the previous statistics:
        if self.previous_statistics is not None:
            data = self.previous_statistics.statistics(data)
        # the first of the statistics need to take list as input, in order to match the API. Then actually the
        # transformations work on np.arrays. In fact the first statistic transforms the list to array. Therefore, the
        # following code needs to be called only if the self statistic is the first, i.e. it does not have a
        # previous_statistic element.
        else:
            data = self._check_and_transform_input(data)

        return np.concatenate((np.mean(data, axis=1).reshape(-1, 1),
                               np.mean(data ** 2, axis=1).reshape(-1, 1)), axis=1)


class TrueSummariesComputationGaussian(nn.Module):
    # this is a torch wrapper of the computation of true summary statistics. I use this for convenience, so that I can
    # use the NeuralEmbedding statistics, even if it may not be super efficient.
    def __init__(self, batch_norm_last_layer=False, affine_batch_norm=True):
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.batch_norm_last_layer = batch_norm_last_layer
        if self.batch_norm_last_layer:
            self.bn_out = nn.BatchNorm1d(2, affine=affine_batch_norm)

    def forward(self, x):  # is this correct, or should I consider the empirical variance instead of mean of x**2?
        output = torch.cat((torch.mean(x, dim=1).view(-1, 1), torch.mean(x ** 2, dim=1).view(-1, 1)), dim=1)
        if self.batch_norm_last_layer:
            output = self.bn_out(output)
        return output


class TrueNaturalParametersComputationGaussian(nn.Module):
    # this is a torch wrapper of the computation of true natural parameters. I use this for convenience, so that I can
    # use the NeuralEmbedding statistics, even if it may not be super efficient.
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):  # TODO this could be optimized by avoiding changing to torch and numpy
        return compute_natural_params_Gaussian(x)

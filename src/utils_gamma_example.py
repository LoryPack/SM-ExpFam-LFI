import numpy as np
import torch
import torch.nn as nn
from abcpy.continuousmodels import ProbabilisticModel, Continuous, InputConnector
from abcpy.statistics import Statistics


# generate one single gamma sample given a theta
def generate_single_gamma_sample(theta, rng=None):
    if rng is None:
        return np.random.gamma(shape=theta[0], scale=theta[1], size=10)
    else:
        return rng.gamma(shape=theta[0], scale=theta[1], size=10)


# generate gamma training examples:

def generate_gamma_training_samples(n_theta=50, size_iid_samples=10, seed=None, k_bounds=[0, 1],
                                    theta_bounds=[0, 1]):
    """This returns a torch tensor."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # this is (k, theta) in the gamma standard parametrization (shape,scale)
    theta_vect = torch.cat((k_bounds[0] + torch.rand(n_theta, 1) * (k_bounds[1] - k_bounds[0]),
                            theta_bounds[0] + torch.rand(n_theta, 1) * (theta_bounds[1] - theta_bounds[0])), 1)

    samples_matrix = np.zeros((n_theta, size_iid_samples))  # n_parameters x R x size_iid_samples
    for i in range(samples_matrix.shape[0]):
        samples_matrix[i, :] = np.random.gamma(shape=theta_vect[i, 0], scale=theta_vect[i, 1],
                                               size=(size_iid_samples))

    samples_matrix = torch.tensor(samples_matrix, requires_grad=False, dtype=torch.float)

    return theta_vect, samples_matrix


def compute_natural_params_Gamma(theta_vect):
    return torch.cat((theta_vect[:, 0].view(-1, 1) - 1, -1 / theta_vect[:, 1].view(-1, 1)), 1)


# THE FOLLOWING THINGS ARE NEEDED FOR ABCpy inference
class IidGamma(ProbabilisticModel, Continuous):
    # this is the ABCpy model for an iid gamma sample.
    def __init__(self, parameters, iid_size=1, name='Iid_Gamma'):

        self.iid_size = iid_size
        input_parameters = InputConnector.from_list(parameters)
        super(IidGamma, self).__init__(input_parameters, name)

    def forward_simulate(self, input_values, num_forward_simulations, rng=np.random.RandomState()):
        k = input_values[0]
        theta = input_values[1]
        result = np.array(rng.gamma(k, theta, (num_forward_simulations, self.iid_size)))
        return [np.array([x]).reshape(-1, ) for x in result]

    def get_output_dimension(self):
        return self.iid_size

    def _check_input(self, input_values):
        """
        Returns True if the standard deviation is negative.
        """
        if len(input_values) != 2:
            return False

        if input_values[1] <= 0 or input_values[0] <= 0:
            return False
        return True

    def _check_output(self, values):
        return all(values > 0)


def extract_params_and_weights_from_journal_gamma(jrnl, step=None):
    params = jrnl.get_parameters(step)
    weights = jrnl.get_weights(step)
    return np.concatenate((np.array(params['k']).reshape(-1, 1), np.array(params['theta']).reshape(-1, 1)),
                          axis=1), weights.reshape(-1)


def extract_params_from_journal_gamma(jrnl, step=None):
    params = jrnl.get_parameters(step)
    return np.concatenate((np.array(params['k']).reshape(-1, 1), np.array(params['theta']).reshape(-1, 1)), axis=1)


def extract_posterior_mean_from_journal_gamma(jrnl, step=None):
    post_mean = jrnl.posterior_mean(step)
    return np.array((post_mean['k'], post_mean['theta']))


class GammaStatistics(Statistics):
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

        return np.concatenate((np.mean(np.log(data), axis=1).reshape(-1, 1), np.mean(data, axis=1).reshape(-1, 1)),
                              axis=1)


class TrueSummariesComputationGamma(nn.Module):
    # this is a torch wrapper of the computation of true summary statistics. I use this for convenience, so that I can
    # use the NeuralEmbedding statistics, even if it may not be super efficient.
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):  # TODO this could be optimized by avoiding changing to torch and numpy
        return torch.from_numpy(np.concatenate((np.mean(np.log(x.detach().numpy()), axis=1).reshape(-1, 1),
                                                np.mean(x.detach().numpy(), axis=1).reshape(-1, 1)),
                                               axis=1).astype("float32"))


class TrueNaturalParametersComputationGamma(nn.Module):
    # this is a torch wrapper of the computation of true summary statistics. I use this for convenience, so that I can
    # use the NeuralEmbedding statistics, even if it may not be super efficient.
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):  # TODO this could be optimized by avoiding changing to torch and numpy
        return compute_natural_params_Gamma(x)

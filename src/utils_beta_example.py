import numpy as np
import torch
import torch.nn as nn
from abcpy.continuousmodels import ProbabilisticModel, Continuous, InputConnector
from abcpy.statistics import Statistics


# generate one single beta sample given a theta
def generate_single_beta_sample(theta, rng=None):
    if rng is None:
        return np.random.beta(theta[0], theta[1], size=10)
    else:
        return rng.beta(theta[0], theta[1], size=10)


# generate beta training examples:

def generate_beta_training_samples(n_theta=50, size_iid_samples=10, seed=None, alpha_bounds=[0, 1],
                                   beta_bounds=[0, 1]):
    # We should avoid $\alpha, \beta < 1$ otherwise we get some samples =0, which give issues in computing the
    # sufficient statistics. This returns a torch tensor.

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # this is (k, theta) in the beta standard parametrization (shape,scale)
    theta_vect = torch.cat((alpha_bounds[0] + torch.rand(n_theta, 1) * (alpha_bounds[1] - alpha_bounds[0]),
                            beta_bounds[0] + torch.rand(n_theta, 1) * (beta_bounds[1] - beta_bounds[0])), 1)

    samples_matrix = np.zeros((n_theta, size_iid_samples))  # n_parameters x R x size_iid_samples
    for i in range(samples_matrix.shape[0]):
        samples_matrix[i, :] = np.random.beta(theta_vect[i, 0], theta_vect[i, 1], size=(size_iid_samples))

    samples_matrix = torch.tensor(samples_matrix, requires_grad=False, dtype=torch.float)

    return theta_vect, samples_matrix


def compute_natural_params_Beta(theta_vect):
    return theta_vect  # the natural parameters of the beta distr are the same as the standard parameters.


# THE FOLLOWING THINGS ARE NEEDED FOR ABCpy inference
class IidBeta(ProbabilisticModel, Continuous):
    # this is the ABCpy model for an iid beta sample.
    def __init__(self, parameters, iid_size=1, name='Iid_Beta'):

        self.iid_size = iid_size
        input_parameters = InputConnector.from_list(parameters)
        super(IidBeta, self).__init__(input_parameters, name)

    def forward_simulate(self, input_values, num_forward_simulations, rng=np.random.RandomState()):
        alpha = input_values[0]
        beta = input_values[1]
        result = np.array(rng.beta(alpha, beta, (num_forward_simulations, self.iid_size)))
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
        return all(values > 0) and all(values < 0)


def extract_params_and_weights_from_journal_beta(jrnl, step=None):
    params = jrnl.get_parameters(step)
    weights = jrnl.get_weights(step)
    return np.concatenate((np.array(params['alpha']).reshape(-1, 1), np.array(params['beta']).reshape(-1, 1)),
                          axis=1), weights.reshape(-1)


def extract_params_from_journal_beta(jrnl, step=None):
    params = jrnl.get_parameters(step)
    return np.concatenate((np.array(params['alpha']).reshape(-1, 1), np.array(params['beta']).reshape(-1, 1)), axis=1)


def extract_posterior_mean_from_journal_beta(jrnl, step=None):
    post_mean = jrnl.posterior_mean(step)
    return np.array((post_mean['alpha'], post_mean['beta']))


class BetaStatistics(Statistics):
    """
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

        return np.concatenate(
            (np.mean(np.log(data), axis=1).reshape(-1, 1), np.mean(np.log(1 - data), axis=1).reshape(-1, 1)), axis=1)


class TrueSummariesComputationBeta(nn.Module):
    # this is a torch wrapper of the computation of true summary statistics. I use this for convenience, so that I can
    # use the NeuralEmbedding statistics, even if it may not be super efficient.
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):  # TODO this could be optimized by avoiding changing to torch and numpy
        return torch.from_numpy(np.concatenate((np.mean(np.log(x.detach().numpy()), axis=1).reshape(-1, 1),
                                                np.mean(np.log(1 - x.detach().numpy()), axis=1).reshape(-1, 1)),
                                               axis=1).astype("float32"))


class TrueNaturalParametersComputationBeta(nn.Module):
    # this is a torch wrapper of the computation of true summary statistics. I use this for convenience, so that I can
    # use the NeuralEmbedding statistics, even if it may not be super efficient.
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):  # TODO this could be optimized by avoiding changing to torch and numpy
        return compute_natural_params_Beta(x)

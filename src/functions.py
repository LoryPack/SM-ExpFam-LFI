import json
import warnings
from functools import reduce
from operator import mul
from time import time

import numpy as np
import ot
import seaborn as sns
import torch
import torch.nn as nn
from abcpy.acceptedparametersmanager import AcceptedParametersManager
from abcpy.distances import Distance
from abcpy.inferences import SABC, APMCABC, RejectionABC, PMCABC, InferenceMethod, SMCABC
from matplotlib import pyplot as plt
from theano import tensor as tt
from tqdm import tqdm


# GENERAL THINGS
def rescale(x):
    x = x - (np.max(x) + np.min(x)) / 2
    return 2 * x / (np.max(x) - np.min(x))


def set_requires_grad(net, value):
    for param in net.parameters():
        param.requires_grad = value


def scale_samples(scaler, samples, requires_grad=True):
    return torch.tensor(scaler.transform(samples.reshape(-1, samples.shape[-1])).astype("float32"),
                        requires_grad=requires_grad).reshape(samples.shape)


def scale_thetas(scaler, thetas, requires_grad=False):
    return torch.tensor(scaler.transform(thetas).astype("float32"), requires_grad=requires_grad)


def jacobian(input, output, diffable=True):
    '''
    Returns the Jacobian matrix (batch x in_size x out_size) of the function that produced the output evaluated at the input

    From https://github.com/mwcvitkovic/MASS-Learning/blob/master/models/utils.py

    Important: need to use diffable=True in order for the training routines based on these to work!

    '''
    assert len(output.shape) == 2
    assert input.shape[0] == output.shape[0]
    in_size = reduce(mul, list(input.shape[1:]), 1)
    if (input.sum() + output.sum()).item() in [np.nan, np.inf]:
        raise ValueError
    J = torch.zeros(list(output.shape) + list(input.shape[1:])).to(input)
    # they are able here to do the gradient computation one batch at a time, of course still considering only one output coordinate at a time
    for i in range(output.shape[1]):
        g = torch.zeros(output.shape).to(input)
        g[:, i] = 1
        if diffable:
            J[:, i] = torch.autograd.grad(output, input, g, only_inputs=True, retain_graph=True, create_graph=True)[0]
        else:
            J[:, i] = torch.autograd.grad(output, input, g, only_inputs=True, retain_graph=True)[0]
    J = J.reshape(output.shape[0], output.shape[1], in_size)
    return J.transpose(2, 1)


def jacobian_second_order(input, output, diffable=True):
    '''
    Returns the Jacobian matrix (batch x in_size x out_size) of the function that produced the output evaluated at the input, as well as
    the matrix of second derivatives of outputs with respect to inputs (batch x in_size x out_size)

    Adapted from https://github.com/mwcvitkovic/MASS-Learning/blob/master/models/utils.py

    Important: need to use diffable=True in order for the training routines based on these to work!
    '''
    assert len(output.shape) == 2
    assert input.shape[0] == output.shape[0]
    in_size = reduce(mul, list(input.shape[1:]), 1)
    if (input.sum() + output.sum()).item() in [np.nan, np.inf]:
        raise ValueError
    J = torch.zeros(list(output.shape) + list(input.shape[1:])).to(input)
    J2 = torch.zeros(list(output.shape) + list(input.shape[1:])).to(input)

    for i in range(output.shape[1]):
        g = torch.zeros(output.shape).to(input)
        g[:, i] = 1
        J[:, i] = torch.autograd.grad(output, input, g, only_inputs=True, retain_graph=True, create_graph=True)[0]
    J = J.reshape(output.shape[0], output.shape[1], in_size)

    for i in range(output.shape[1]):
        for j in range(input.shape[1]):
            g = torch.zeros(J.shape).to(input)
            g[:, i, j] = 1
            if diffable:
                J2[:, i, j] = torch.autograd.grad(J, input, g, only_inputs=True, retain_graph=True, create_graph=True)[
                                  0][:, j]
            else:
                J2[:, i, j] = torch.autograd.grad(J, input, g, only_inputs=True, retain_graph=True)[0][:, j]

    J2 = J2.reshape(output.shape[0], output.shape[1], in_size)

    return J.transpose(2, 1), J2.transpose(2, 1)


def jacobian_hessian(input, output, diffable=True):
    '''
    Returns the Jacobian matrix (batch x in_size x out_size) of the function that produced the output evaluated at the input, as well as
    the Hessian matrix (batch x in_size x in_size x out_size).

    This takes slightly more than the jacobian_second_order routine.

    Adapted from https://github.com/mwcvitkovic/MASS-Learning/blob/master/models/utils.py

    Important: need to use diffable=True in order for the training routines based on these to work!
    '''
    assert len(output.shape) == 2
    assert input.shape[0] == output.shape[0]
    in_size = reduce(mul, list(input.shape[1:]), 1)
    if (input.sum() + output.sum()).item() in [np.nan, np.inf]:
        raise ValueError
    J = torch.zeros(list(output.shape) + list(input.shape[1:])).to(input)
    H = torch.zeros(list(output.shape) + list(input.shape[1:]) + list(input.shape[1:])).to(input)

    for i in range(output.shape[1]):
        g = torch.zeros(output.shape).to(input)
        g[:, i] = 1
        J[:, i] = torch.autograd.grad(output, input, g, only_inputs=True, retain_graph=True, create_graph=True)[0]
    J = J.reshape(output.shape[0], output.shape[1], in_size)

    for i in range(output.shape[1]):
        for j in range(input.shape[1]):
            g = torch.zeros(J.shape).to(input)
            g[:, i, j] = 1
            if diffable:
                H[:, i, j] = torch.autograd.grad(J, input, g, only_inputs=True, retain_graph=True, create_graph=True)[0]
            else:
                H[:, i, j] = torch.autograd.grad(J, input, g, only_inputs=True, retain_graph=True)[0]

    return J.transpose(2, 1), H.transpose(3, 1)


class DummyScaler():
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x if isinstance(x, np.ndarray) else x.numpy()

    def inverse_transform(self, x):
        return x if isinstance(x, np.ndarray) else x.numpy()


# UTILITIES FOR INFERENCE:
class RescaleAndNet(nn.Module):
    # define an architecture that applies rescaling first and then applies the nn:
    def __init__(self, net, scaler):
        super().__init__()
        self.net = net
        self.scaler = scaler

    def forward(self, x):
        x = scale_samples(self.scaler, x, requires_grad=False)
        return self.net(x)


class DiscardLastOutputNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        # this is not very elegant; by using Python, it could be done better with np.take,
        # but there is no analogue in Pytorch.
        if len(x.shape) == 1:
            return self.net(x)[0:-1]
        if len(x.shape) == 2:
            return self.net(x)[:, 0:-1]
        if len(x.shape) == 3:
            return self.net(x)[:, :, 0:-1]


class RescaleAndDiscardLastOutputNet(nn.Module):
    # define an architecture that applies rescaling first and then applies the nn:
    def __init__(self, net, scaler):
        super().__init__()
        self.net = net
        self.scaler = scaler

    def forward(self, x):
        x = scale_samples(self.scaler, x, requires_grad=False)
        if len(x.shape) == 1:
            return self.net(x)[0:-1]
        if len(x.shape) == 2:
            return self.net(x)[:, 0:-1]
        if len(x.shape) == 3:
            return self.net(x)[:, :, 0:-1]


def determine_eps(samples_matrix, dist_calc, quantile):
    dist = []
    for i in range(len(samples_matrix)):
        for j in range(i + 1, len(samples_matrix)):
            dist.append(dist_calc.distance([samples_matrix[i].detach().numpy()], [samples_matrix[j].detach().numpy()]))
    return np.quantile(dist, quantile)


def ABC_inference(algorithm, model, observation, distance_calculator, eps, n_samples, n_steps, backend, seed=None,
                  full_output=0, **kwargs):
    """NB: eps represents initial value of epsilon for PMCABC and SABC; represents the single eps value for RejectionABC
     and represents the final value for SMCABC."""
    start = time()
    if algorithm == "PMCABC":
        sampler = PMCABC([model], [distance_calculator], backend, seed=seed)
        jrnl = sampler.sample([[observation]], n_steps, np.array([eps]), n_samples=n_samples, full_output=full_output,
                              **kwargs)
    if algorithm == "APMCABC":
        sampler = APMCABC([model], [distance_calculator], backend, seed=seed)
        jrnl = sampler.sample([[observation]], n_steps, n_samples=n_samples, full_output=full_output, **kwargs)
    elif algorithm == "SABC":
        sampler = SABC([model], [distance_calculator], backend, seed=seed)
        jrnl = sampler.sample([[observation]], n_steps, eps, n_samples=n_samples, full_output=full_output, **kwargs)
    elif algorithm == "RejectionABC":
        sampler = RejectionABC([model], [distance_calculator], backend, seed=seed)
        jrnl = sampler.sample([[observation]], eps, n_samples=n_samples, full_output=full_output, **kwargs)
    elif algorithm == "SMCABC":
        # for this usually a larger number of steps is required. alpha can be left to 0.95, covFactor=2 and
        # resample=None. epsilon_final is instead important to fix!
        sampler = SMCABC([model], [distance_calculator], backend, seed=seed)
        # sample(observations, steps, n_samples=10000, n_samples_per_param=1, epsilon_final=0.1, alpha=0.95,
        #        covFactor=2, resample=None, full_output=0, which_mcmc_kernel=0, journal_file=None)
        jrnl = sampler.sample([[observation]], n_steps, n_samples=n_samples, full_output=full_output, epsilon_final=eps,
                              **kwargs)

    print("It took ", time() - start, " seconds.")

    return jrnl


class WeightedEuclidean(Distance):
    """
    This class implements weighted Euclidean distance between two vectors. It standardizes the different statistics
    using their standard deviation.

    The maximum value of the distance is np.inf.
    """

    def __init__(self, statistics, training_data):
        # training data is used to find the standard deviations of the distances
        self.statistics_calc = statistics

        training_statistics = self.statistics_calc.statistics(training_data)
        self.std_statistics = np.std(training_statistics, axis=0)  # we store this and use it over time

        if np.any(np.isinf(self.std_statistics)):
            warnings.warn("Infinity in the standard deviation", RuntimeWarning)
        if np.any(np.isnan(self.std_statistics)):
            warnings.warn("nan in the  standard deviation", RuntimeWarning)

        # Since the observations do always stay the same, we can save the
        #  summary statistics of them and not recalculate it each time
        self.s1 = None
        self.data_set = None
        self.dataSame = False

    def distance(self, d1, d2):
        """Calculates the distance between two datasets.

        Parameters
        ----------
        d1, d2: list
            A list, containing a list describing the data set
        """

        if not isinstance(d1, list):
            raise TypeError('Data is not of allowed types')
        if not isinstance(d2, list):
            raise TypeError('Data is not of allowed types')

        # Check whether d1 is same as self.data_set
        if self.data_set is not None:
            if len(np.array(d1[0]).reshape(-1, )) == 1:
                self.data_set == d1
            else:
                self.dataSame = all([(np.array(self.data_set[i]) == np.array(d1[i])).all() for i in range(len(d1))])

        # Extract summary statistics from the dataset
        if (self.s1 is None or self.dataSame is False):
            self.s1 = self.statistics_calc.statistics(d1) / self.std_statistics
            self.data_set = d1

        s2 = self.statistics_calc.statistics(d2) / self.std_statistics

        # compute distance between the statistics
        dist = np.zeros(shape=(self.s1.shape[0], s2.shape[0]))
        for ind1 in range(0, self.s1.shape[0]):
            for ind2 in range(0, s2.shape[0]):
                dist[ind1, ind2] = np.sqrt(np.sum(pow(self.s1[ind1, :] - s2[ind2, :], 2)))

        return dist.mean()

    def dist_max(self):
        return np.inf


def generate_training_samples_ABC_model(model, k, seed=None):
    """Quite inefficient as it uses a loop to generate the k observations. It works also if one of the parameters of the
    model is an hyperparameter.
    Model has to implement a `get_number_parameters` and `get_output_dimension` method, in order to define the correct arrays
     (this may actually not be needed, I could work with lists maybe, or stacking the output arrays).
     Also, here this works only for models whose output is a 1d array for now."""

    rng = np.random.RandomState(seed=seed)
    simulations = np.zeros((k, model.get_output_dimension()))
    parameters = np.zeros((k, model.get_number_parameters()))

    for i in tqdm(range(k)):

        # we sample from the prior for the parameters of the model
        params = []
        for parent in model.get_input_models():
            # each parameter of the model is specified by a distribution with some fixed hyperparameters; the following gets these hyperparameters
            param_list_parent = []
            for hyperparam in parent.get_input_models():
                param_list_parent.append(hyperparam.forward_simulate([], 1, rng=rng)[0])
            # print(param_list_parent)

            params.append(parent.forward_simulate(param_list_parent, 1, rng=rng)[0])

            # print(parent.get_input_models().forward_simulate())
        # print(params)
        parameters[i] = np.array(params).reshape(-1)
        simulations[i] = model.forward_simulate(parameters[i], 1, rng=rng)[0]

    return parameters, simulations


class LogLike(tt.Op):
    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, observation):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.observation = observation

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.observation)

        outputs[0][0] = np.array(logl)  # output the log-likelihood


# RESULT ANALYSIS:

def subsample_trace(trace, size=1000):
    if len(trace) < size:
        return trace
    return trace[np.random.choice(range(len(trace)), size=size, replace=False)]


def subsample_trace_and_weights(trace, weights, size=1000):
    if len(trace) < size:
        return trace, weights
    indeces = np.random.choice(range(len(trace)), size=size, replace=False)
    return trace[indeces], weights[indeces]


def wass_dist(post_samples_1, post_samples_2, weights_post_1=None, weights_post_2=None, numItermax=100000):
    """Computes the Wasserstein 2 distance.

    post_samples_1 and post_post_samples_2 are 2 dimensional arrays: first dim is the number of samples, 2nd dim is the
    number of coordinates in the each sample.

    We allow to give weights to the posterior distribution. Leave weights_post_1 and weights_post_2 to None if your
    samples do not have weights. """

    n = post_samples_1.shape[0]
    m = post_samples_2.shape[0]

    if weights_post_1 is None:
        a = np.ones((n,)) / n
    else:
        if len(weights_post_1) != n:
            raise RuntimeError("Number of weights and number of samples need to be the same.")
        a = weights_post_1 / np.sum(weights_post_1)
    if weights_post_2 is None:
        b = np.ones((m,)) / m
    else:
        if len(weights_post_2) != m:
            raise RuntimeError("Number of weights and number of samples need to be the same.")
        b = weights_post_2 / np.sum(weights_post_2)

    # loss matrix
    M = ot.dist(x1=post_samples_1, x2=post_samples_2)  # this returns squared distance!
    # can use the following to return directly the cost:
    cost = ot.emd2(a, b, M, numItermax=numItermax)

    return np.sqrt(cost)


def plot_losses(loss_list, test_loss_list, file_name):
    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    ax[0].plot(loss_list)
    ax[1].plot(test_loss_list)
    ax[0].set_title("Training loss")
    ax[1].set_title("Test loss")
    ax[0].set_xlabel("Training iteration")
    ax[1].set_xlabel("Training iteration")
    plt.savefig(file_name)
    plt.close()


def plot_single_marginal_with_trace_samples(theta_obs, trace_approx, trace_true=None, weights_trace_approx=None,
                                            param_names=[], namefile=None, thetarange=None, ):
    if theta_obs is not None:
        n_plots = len(theta_obs)
    else:
        n_plots = trace_approx.shape[1]
    # produce single marginal plot
    fig, ax = plt.subplots(1, n_plots, figsize=(9 * n_plots, 9))
    if n_plots == 1:
        ax = [ax]

    if trace_true is not None:
        true_post_means = np.mean(trace_true, axis=0)
        for i in range(n_plots):
            sns.kdeplot(trace_true[:, i], ax=ax[i], color="C0", label="True posterior")
            ax[i].axvline(true_post_means[i], color="C0", ls=":")

    approx_post_means = np.mean(trace_approx, axis=0)
    for i in range(n_plots):
        sns.kdeplot(trace_approx[:, i], weights=weights_trace_approx, ax=ax[i], color="C4",
                    label="Approximate posterior")
        ax[i].axvline(approx_post_means[i], color="C4", ls=":")

        if theta_obs is not None:
            ax[i].axvline(theta_obs[i], color="green", label="True value")
        ax[i].legend()

        if len(param_names) != 0:
            ax[i].set_title(param_names[i])

        if thetarange is not None:
            ax[i].set_xlim(thetarange[0, i], thetarange[1, i])

    if namefile is not None:
        plt.savefig(namefile, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    return fig, ax


def plot_bivariate_marginal_with_trace_samples(theta_obs, trace_approx, trace_true=None, trace_approx_2=None,
                                               weights_trace_approx=None, weights_trace_approx_2=None, thetarange=None,
                                               param1_name=r"$\mu$",
                                               param2_name=r"$\sigma$", namefile=None, color="C0", figsize_vertical=9,
                                               legend=True, title_size=None, label_size=None, space_subplots=None,
                                               vertical=False, thresh=0.05):
    # produce single marginal plot
    number_techniques_to_plot = 1 + (0 if trace_true is None else 1) + (0 if trace_approx_2 is None else 1)
    if vertical:
        fig, ax = plt.subplots(number_techniques_to_plot, 1,
                               figsize=(figsize_vertical, figsize_vertical * number_techniques_to_plot))
        fig.subplots_adjust(hspace=space_subplots)
    else:
        fig, ax = plt.subplots(1, number_techniques_to_plot, figsize=(
            figsize_vertical * number_techniques_to_plot + 2 * (number_techniques_to_plot - 1) * (
                0.2 if space_subplots is None else space_subplots), figsize_vertical))
        fig.subplots_adjust(wspace=space_subplots)
    if number_techniques_to_plot == 1:
        ax = [ax]

    ax_index = 0
    if trace_true is not None:
        true_post_means = np.mean(trace_true, axis=0)
        sns.kdeplot(x=trace_true[:, 0], y=trace_true[:, 1], color=color, shade=True, thresh=thresh, alpha=1,
                    ax=ax[ax_index])
        ax[ax_index].axvline(true_post_means[0], color=color, ls=":", label="Posterior mean")
        ax[ax_index].axhline(true_post_means[1], color=color, ls=":")
        ax[ax_index].set_title("True posterior", size=title_size)
        ax_index += 1

    approx_post_means = np.mean(trace_approx, axis=0)
    sns.kdeplot(x=trace_approx[:, 0], y=trace_approx[:, 1], color=color, shade=True, thresh=thresh, alpha=1,
                ax=ax[ax_index], weights=weights_trace_approx, )
    ax[ax_index].axvline(approx_post_means[0], color=color, ls=":", label="Posterior mean")
    ax[ax_index].axhline(approx_post_means[1], color=color, ls=":")
    ax[ax_index].set_title("Approximate posterior", size=title_size)
    ax_index += 1

    if trace_approx_2 is not None:
        approx_post_means_2 = np.mean(trace_approx_2, axis=0)
        sns.kdeplot(x=trace_approx_2[:, 0], y=trace_approx_2[:, 1], color=color, shade=True, thresh=thresh, alpha=1,
                    ax=ax[ax_index], weights=weights_trace_approx_2, )
        ax[ax_index].axvline(approx_post_means_2[0], color=color, ls=":", label="Posterior mean")
        ax[ax_index].axhline(approx_post_means_2[1], color=color, ls=":")
        ax[ax_index].set_title("Approx posterior 2", size=title_size)
        ax_index += 1

    # ax[0].axvline(theta_obs[0], color="green", label="True value")
    # ax[1].axvline(theta_obs[1], color="green", label="True value")

    for ax_index in range(number_techniques_to_plot):
        if theta_obs is not None:
            ax[ax_index].axvline(theta_obs[0], color="green", label="True value")
            ax[ax_index].axhline(theta_obs[1], color="green")
        ax[ax_index].set_xlabel(param1_name, size=label_size)
        ax[ax_index].set_ylabel(param2_name, size=label_size)
        if thetarange is not None:
            ax[ax_index].set_xlim(thetarange[0, 0], thetarange[1, 0])
            ax[ax_index].set_ylim(thetarange[0, 1], thetarange[1, 1])
        if legend:
            ax[ax_index].legend()

    if namefile is not None:
        plt.savefig(namefile, bbox_inches="tight")
        # else:
        #     plt.show()
        plt.close()

    return fig, ax


def plot_trace(trace, theta_dim, param_name=None, namefile=None, burnin=None):
    fig, ax = plt.subplots(ncols=theta_dim, nrows=1, figsize=(theta_dim * 4, 4))
    for i in range(theta_dim):
        ax[i].plot(trace[:, i])
        ax[i].set_title(param_name[i])
        if burnin is not None:
            ax[i].axvline(burnin, color="red", ls="dotted", alpha=0.5)
    if namefile is not None:
        plt.savefig(namefile, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def estimate_autocorrelation_time(traces, theta_dim, param_names, textfile=None):
    # we estimate here the integrated autocorrelation time; see https://dfm.io/posts/autocorr/
    def next_pow_two(n):
        i = 1
        while i < n:
            i = i << 1
        return i

    def autocorr_func_1d(x, norm=True):
        x = np.atleast_1d(x)
        if len(x.shape) != 1:
            raise ValueError("invalid dimensions for 1D autocorrelation function")
        n = next_pow_two(len(x))

        # Compute the FFT and then (from that) the auto-correlation function
        f = np.fft.fft(x - np.mean(x), n=2 * n)
        acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
        acf /= 4 * n

        # Optionally normalize
        if norm:
            acf /= acf[0]

        return acf

    # Automated windowing procedure following Sokal (1989)
    def auto_window(taus, c):
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    def autocorr_new(y, c=5.0):
        f = np.zeros(y.shape[1])
        for yy in y:
            f += autocorr_func_1d(yy)
        f /= len(y)
        taus = 2.0 * np.cumsum(f) - 1.0
        window = auto_window(taus, c)
        return taus[window]

    # traces = np.zeros((n_samples, n_observations - start_observation_index, theta_dim))

    autocorr_time = np.zeros((traces.shape[1], traces.shape[2]))
    for j in range(traces.shape[1]):
        for i in range(theta_dim):
            autocorr_time[j, i] = autocorr_new(traces[:, j, i].reshape(1, -1))

    string = map(lambda x: x.ljust(16), param_names, )
    string = reduce((lambda x, y: x + y), string)
    if textfile is not None:
        text_file = open(textfile, "w")
        text_file.write(string + "\n")
    print(string)
    for j in range(traces.shape[1]):
        string = map(lambda x: f"{x:.4f}".ljust(16), autocorr_time[j], )
        string = reduce((lambda x, y: x + y), string)
        print(string)
        if textfile is not None:
            text_file.write(string + "\n")
    if textfile is not None:
        text_file.close()


def plot_confidence_bands_performance_vs_iteration(data, start_step=0, end_step=None, fig=None, ax=None, band_1=25,
                                                   band_2=75, band_3=95, alpha_1=0.7, alpha_2=0.5, alpha_3=0.3,
                                                   color_band_1="C1", color_band_2="C1", color_band_3="C1",
                                                   color_line="C3", ls_band_1='--', ls_band_2='--', ls_band_3='--',
                                                   fill_between=True, hatch=None, **kwargs):
    # data is a 2 dimensional array, 1st dimension is the iteration and 2nd is the number of observation.
    # this plots the median and the confidence intervals of the corresponding size.

    if end_step is None:
        end_step = data.shape[0]

    median_simulation = np.median(data, axis=1)  # this takes the pointwise median
    # mean_simulation = np.mean(data, axis=1)  # this takes the pointwise mean
    lower_simulation_1 = np.percentile(data, 50 - band_1 / 2, axis=1)
    upper_simulation_1 = np.percentile(data, 50 + band_1 / 2, axis=1)
    lower_simulation_2 = np.percentile(data, 50 - band_2 / 2, axis=1)
    upper_simulation_2 = np.percentile(data, 50 + band_2 / 2, axis=1)
    lower_simulation_3 = np.percentile(data, 50 - band_3 / 2, axis=1)
    upper_simulation_3 = np.percentile(data, 50 + band_3 / 2, axis=1)

    if fig is None or ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))

    ax.plot(np.arange(start_step, end_step), median_simulation[start_step:end_step], color=color_line,
            alpha=1, **kwargs)
    # ax.plot(np.arange(start_step, end_step), mean_simulation[start_step:end_step], color=color_line,
    #         alpha=alpha, **kwargs)
    if fill_between:
        ax.fill_between(np.arange(start_step, end_step), lower_simulation_1[start_step:end_step],
                        upper_simulation_1[start_step:end_step], alpha=alpha_1,
                        facecolor=color_band_1 if hatch is None else "none", hatch=hatch,
                        edgecolor=color_band_1 if hatch is not None else None)
        ax.fill_between(np.arange(start_step, end_step), lower_simulation_2[start_step:end_step],
                        upper_simulation_2[start_step:end_step], alpha=alpha_2, facecolor=color_band_2,
                        edgecolor=color_band_2)
        ax.fill_between(np.arange(start_step, end_step), lower_simulation_3[start_step:end_step],
                        upper_simulation_3[start_step:end_step], alpha=alpha_3, facecolor=color_band_3,
                        edgecolor=color_band_3)
    else:
        ax.plot(np.arange(start_step, end_step), lower_simulation_1[start_step:end_step],
                alpha=alpha_1, color=color_band_1, ls=ls_band_1)
        ax.plot(np.arange(start_step, end_step), upper_simulation_1[start_step:end_step],
                alpha=alpha_1, color=color_band_1, ls=ls_band_1)
        ax.plot(np.arange(start_step, end_step), lower_simulation_2[start_step:end_step],
                alpha=alpha_2, color=color_band_2, ls=ls_band_2)
        ax.plot(np.arange(start_step, end_step), upper_simulation_2[start_step:end_step],
                alpha=alpha_2, color=color_band_2, ls=ls_band_2)
        ax.plot(np.arange(start_step, end_step), lower_simulation_3[start_step:end_step],
                alpha=alpha_3, color=color_band_3, ls=ls_band_3)
        ax.plot(np.arange(start_step, end_step), upper_simulation_3[start_step:end_step],
                alpha=alpha_3, color=color_band_3, ls=ls_band_3)
    return fig, ax


def check_iteration_better_perf(iterative_method_wass_dist, reference_wass_dist):
    # iterative_method_wass_dist has shape n_steps x n_observations
    iterative_method_wass_dist_median = np.median(iterative_method_wass_dist, axis=1)
    reference_wass_dist_median = np.median(reference_wass_dist)
    boolean_array = iterative_method_wass_dist_median < reference_wass_dist_median
    if np.logical_not(boolean_array).all():  # all of them are False:
        val = None
    else:
        val = np.argmax(iterative_method_wass_dist_median < reference_wass_dist_median)

    return val


def save_dict_to_json(dictionary, file):
    with open(file, 'w') as fp:
        json.dump(dictionary, fp, sort_keys=True, indent=4)


def try_loading_wass_RMSE_post_mean(folder, filename_postfix, load_flag):
    wass_dist = None
    RMSE_post_mean = None
    if load_flag:
        try:
            wass_dist = np.load(folder + "wass_dist" + filename_postfix + ".npy")
            RMSE_post_mean = np.load(folder + "RMSE_post_mean" + filename_postfix + ".npy")
            load_successful = True
            print(f"Data from {folder} was loaded successfully.")
        except FileNotFoundError:
            print(f"Data from {folder} was not loaded successfully.")
            load_successful = False
    else:
        load_successful = False
    return wass_dist, RMSE_post_mean, load_successful


class DrawFromParamValues(InferenceMethod):
    model = None
    rng = None
    n_samples = None
    backend = None

    n_samples_per_param = None  # this needs to be there otherwise it does not instantiate correctly

    def __init__(self, root_models, backend, seed=None, discard_too_large_values=True):
        self.model = root_models
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.discard_too_large_values = discard_too_large_values
        # An object managing the bds objects
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)

    def sample(self, param_values):

        self.param_values = param_values  # list of parameter values
        self.n_samples = len(param_values)
        self.accepted_parameters_manager.broadcast(self.backend, 1)

        # now generate an array of seeds that need to be different one from the other. One way to do it is the
        # following.
        # Moreover, you cannot use int64 as seeds need to be < 2**32 - 1. How to fix this?
        # Note that this is not perfect; you still have small possibility of having some seeds that are equal. Is there
        # a better way? This would likely not change much the performance
        # An idea would be to use rng.choice but that is too
        seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=self.n_samples, dtype=np.uint32)
        # check how many equal seeds there are and remove them:
        sorted_seed_arr = np.sort(seed_arr)
        indices = sorted_seed_arr[:-1] == sorted_seed_arr[1:]
        # print("Number of equal seeds:", np.sum(indices))
        if np.sum(indices) > 0:
            # the following can be used to remove the equal seeds in case there are some
            sorted_seed_arr[:-1][indices] = sorted_seed_arr[:-1][indices] + 1
        # print("Number of equal seeds after update:", np.sum(sorted_seed_arr[:-1] == sorted_seed_arr[1:]))
        rng_arr = np.array([np.random.RandomState(seed) for seed in sorted_seed_arr])
        # zip with the param values:
        data_arr = list(zip(self.param_values, rng_arr))
        data_pds = self.backend.parallelize(data_arr)

        parameters_simulations_pds = self.backend.map(self._sample_parameter, data_pds)
        parameters_simulations = self.backend.collect(parameters_simulations_pds)
        parameters, simulations = [list(t) for t in zip(*parameters_simulations)]

        parameters = np.array(parameters).squeeze()
        simulations = np.array(simulations).squeeze()

        return parameters, simulations

    def _sample_parameter(self, data, npc=None):
        theta, rng = data[0], data[1]

        ok_flag = False

        while not ok_flag:
            # assume that we have one single model
            y_sim = self.model[0].forward_simulate(theta, 1, rng=rng)
            # self.sample_from_prior(rng=rng)
            # theta = self.get_parameters(self.model)
            # y_sim = self.simulate(1, rng=rng, npc=npc)

            # if there are no potential infinities there (or if we do not check for those).
            # For instance, Lorenz model may give too large values sometimes (quite rarely).
            if np.sum(np.isinf(np.array(y_sim).astype("float32"))) > 0 and self.discard_too_large_values:
                print("y_sim contained too large values for float32; simulating again.")
            else:
                ok_flag = True

        return theta, y_sim

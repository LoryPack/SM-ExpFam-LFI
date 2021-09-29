import numpy as np

from abcpy.inferences import DrawFromPrior


class EnergyScore:
    """ Estimates the EnergyScore. Here, I assume the observations and simulations are arrays of size (n_obs, p) and
    (n_sim, p), p being the dimensionality.
    Then, for each fixed observation the n_sim simulations are used to estimate the
    scoring rule. Subsequently, the values are summed over each of the n_obs observations.

    Note this scoring rule is connected to the energy distance between probability distributions.
    """

    def __init__(self, beta=1):
        """default value is beta=1"""
        self.beta = beta
        self.beta_over_2 = 0.5 * beta

    def score(self, observations, simulations):
        score = self.estimate_energy_score_new(observations, simulations)
        return score

    def estimate_energy_score_new(self, observations, simulations):
        """observations is an array of size (n_obs, p) (p being the dimensionality), while simulations is an array
        of size (n_sim, p).
        We estimate this by building an empirical unbiased estimate of Eq. (2) in Ziel and Berk 2019"""
        n_obs = observations.shape[0]
        n_sim, p = simulations.shape
        diff_X_y = observations.reshape(n_obs, 1, -1) - simulations.reshape(1, n_sim, p)
        # check (specifically in case n_sim==p):
        # diff_X_y2 = np.zeros((observations.shape[0], *simulations.shape))
        # for i in range(observations.shape[0]):
        #     for j in range(n_sim):
        #         diff_X_y2[i, j] = observations[i] - simulations[j]
        # assert np.allclose(diff_X_y2, diff_X_y)
        diff_X_y = np.einsum('ijk, ijk -> ij', diff_X_y, diff_X_y)

        diff_X_tildeX = simulations.reshape(1, n_sim, p) - simulations.reshape(n_sim, 1, p)
        # check (specifically in case n_sim==p):
        # diff_X_tildeX2 = np.zeros((n_sim, n_sim, p))
        # for i in range(n_sim):
        #     for j in range(n_sim):
        #         diff_X_tildeX2[i, j] = simulations[j] - simulations[i]
        # assert np.allclose(diff_X_tildeX2, diff_X_tildeX)
        diff_X_tildeX = np.einsum('ijk, ijk -> ij', diff_X_tildeX, diff_X_tildeX)

        if self.beta_over_2 != 1:
            diff_X_y **= self.beta_over_2
            diff_X_tildeX **= self.beta_over_2

        return 2 * np.sum(np.mean(diff_X_y, axis=1)) - n_obs * np.sum(diff_X_tildeX) / (n_sim * (n_sim - 1))
        # here I am using an unbiased estimate; I could also use a biased estimate (dividing by n_sim**2). In the ABC
        # with energy distance, they use the biased estimate for energy distance as it is always positive; not sure this
        # is so important here however.


class KernelScore:
    """ Estimates the KernelScore. Here, I assume the observations and simulations are arrays of size (n_obs, p) and
    (n_sim, p), p being the dimensionality.
    Then, for each fixed observation the n_sim simulations are used to estimate the
    scoring rule. Subsequently, the values are summed over each of the n_obs observations.

    Note this scoring rule is connected to the MMD between probability distributions.
    """

    def __init__(self, kernel="gaussian", biased_estimator=False, **kernel_kwargs):
        """
        Parameters
        ----------
        kernel : str or callable, optional
            Can be a string denoting the kernel, or a function. If a string, only gaussian is implemented for now; in
            that case, you can also provide an additional keyword parameter 'sigma' which is used as the sigma in the
            kernel.
        """

        self.kernel_vectorized = False
        if not isinstance(kernel, str) and not callable(kernel):
            raise RuntimeError("'kernel' must be either a string or a function of two variables returning a scalar.")
        if isinstance(kernel, str):
            if kernel == "gaussian":
                self.kernel = self.def_gaussian_kernel(**kernel_kwargs)
                self.kernel_vectorized = True  # the gaussian kernel is vectorized
            else:
                raise NotImplementedError("The required kernel is not implemented.")
        else:
            self.kernel = kernel  # if kernel is a callable already

        self.biased_estimator = biased_estimator

    def score(self, observations, simulations):
        # compute the Gram matrix
        K_sim_sim, K_obs_sim = self.compute_Gram_matrix(observations, simulations)

        # Estimate MMD
        if self.biased_estimator:
            return self.MMD_V_estimator(K_sim_sim, K_obs_sim)
        else:
            return self.MMD_unbiased(K_sim_sim, K_obs_sim)

    @staticmethod
    def def_gaussian_kernel(sigma=1):
        # notice in the MMD paper they set sigma to a median value over the observation; check that.
        sigma_2 = 2 * sigma ** 2

        # def Gaussian_kernel(x, y):
        #     xy = x - y
        #     # assert np.allclose(np.dot(xy, xy), np.linalg.norm(xy) ** 2)
        #     return np.exp(- np.dot(xy, xy) / sigma_2)

        def Gaussian_kernel_vectorized(X, Y):
            """Here X and Y have shape (n_samples_x, n_features) and (n_samples_y, n_features);
            this directly computes the kernel for all pairwise components"""
            XY = X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1)  # pairwise differences
            return np.exp(- np.einsum('xyi,xyi->xy', XY, XY) / sigma_2)

        return Gaussian_kernel_vectorized

    def compute_Gram_matrix(self, s_observations, s_simulations):

        if self.kernel_vectorized:
            K_sim_sim = self.kernel(s_simulations, s_simulations)
            K_obs_sim = self.kernel(s_observations, s_simulations)
        else:
            n_obs = s_observations.shape[0]
            n_sim = s_simulations.shape[0]

            K_sim_sim = np.zeros((n_sim, n_sim))
            K_obs_sim = np.zeros((n_obs, n_sim))

            for i in range(n_sim):
                # we assume the function to be symmetric; this saves some steps:
                for j in range(i, n_sim):
                    K_sim_sim[j, i] = K_sim_sim[i, j] = self.kernel(s_simulations[i], s_simulations[j])

            for i in range(n_obs):
                for j in range(n_sim):
                    K_obs_sim[i, j] = self.kernel(s_observations[i], s_simulations[j])

        return K_sim_sim, K_obs_sim

    @staticmethod
    def MMD_unbiased(K_sim_sim, K_obs_sim):
        # Adapted from https://github.com/eugenium/MMD/blob/2fe67cbc7378f10f3b273cfd8d8bbd2135db5798/mmd.py
        # The estimate when distribution of x is not equal to y
        n_obs, n_sim = K_obs_sim.shape

        t_obs_sim = (2. / n_sim) * np.sum(K_obs_sim)
        t_sim_sim = (1. / (n_sim * (n_sim - 1))) * np.sum(K_sim_sim - np.diag(np.diagonal(K_sim_sim)))

        return n_obs * t_sim_sim - t_obs_sim

    @staticmethod
    def MMD_V_estimator(K_sim_sim, K_obs_sim):
        # The estimate when distribution of x is not equal to y
        n_obs, n_sim = K_obs_sim.shape

        t_obs_sim = (2. / n_sim) * np.sum(K_obs_sim)
        t_sim_sim = (1. / (n_sim * n_sim)) * np.sum(K_sim_sim)

        return n_obs * t_sim_sim - t_obs_sim


def _estimate_score_timeseries(simulations_timeseries, observation_timeseries, score="kernel", **kwargs):
    """Here, simulations_timeseries is of shape (n_sim, p, timesteps) and observation_timeseries is of shape
    (p, timesteps)"""

    assert len(simulations_timeseries.shape) == 3
    assert len(observation_timeseries.shape) == 2
    assert simulations_timeseries.shape[1] == observation_timeseries.shape[0]
    assert simulations_timeseries.shape[2] == observation_timeseries.shape[1]

    n_timesteps = simulations_timeseries.shape[2]

    sr_values = np.zeros(n_timesteps)
    if score == "kernel":
        scoring_rule = KernelScore(**kwargs)
    elif score == "energy":
        scoring_rule = EnergyScore(**kwargs)
    else:
        raise RuntimeError

    for t in range(n_timesteps):
        sr_values[t] = scoring_rule.score(observation_timeseries[:, t], simulations_timeseries[:, :, t])

    return sr_values, np.sum(sr_values)


def estimate_kernel_score_timeseries(simulations_timeseries, observation_timeseries, **kwargs):
    return _estimate_score_timeseries(simulations_timeseries, observation_timeseries, score="kernel", **kwargs)


def estimate_energy_score_timeseries(simulations_timeseries, observation_timeseries, **kwargs):
    return _estimate_score_timeseries(simulations_timeseries, observation_timeseries, score="energy", **kwargs)


def estimate_bandwidth_timeseries(model_abc, backend, num_vars, n_theta=100, seed=42, return_values=["median"]):
    """Estimate the bandwidth for the gaussian kernel in KernelSR. Specifically, it generates n_samples_per_param
    simulations for each theta, then computes the pairwise distances and takes the median of it. The returned value
    is the median (by default; you can also compute the mean if preferred) of the latter over all considered values
    of theta.  """

    # generate the values of theta from prior
    theta_vect, simulations_theta_vect = DrawFromPrior([model_abc], backend, seed=seed).sample_par_sim_pairs(n_theta, 1)
    simulations_theta_vect = simulations_theta_vect.reshape(n_theta, num_vars, -1)  # last index is the timestep
    n_timestep = simulations_theta_vect.shape[2]

    distances_median = np.zeros(n_timestep)
    for timestep_index in range(n_timestep):
        simulations = simulations_theta_vect[:, :, timestep_index]
        distances = np.linalg.norm(
            simulations.reshape(1, n_theta, -1) - simulations.reshape(n_theta, 1, -1), axis=-1)[
            ~np.eye(n_theta, dtype=bool)].reshape(-1)
        # take the median over the second index:
        distances_median[timestep_index] = np.median(distances)

    return_list = []
    if "median" in return_values:
        return_list.append(np.median(distances_median.flatten()))
    if "mean" in return_values:
        return_list.append(np.mean(distances_median.flatten()))

    return return_list[0] if len(return_list) == 1 else return_list

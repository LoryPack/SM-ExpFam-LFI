import os
import sys
from time import sleep

import numpy as np
import pymc3 as pm
import theano.tensor as tt

sys.path.append(os.getcwd())  # for some reason it does not see my files if I don't put this

from abcpy.continuousmodels import Uniform

from src.utils_gaussian_example import generate_gaussian_training_samples
from src.utils_gamma_example import generate_gamma_training_samples
from src.utils_beta_example import generate_beta_training_samples
from src.utils_arma_example import ARMAmodel, ar2_log_lik_for_mcmc, ma2_log_lik_for_mcmc
from src.utils_Lorenz95_example import StochLorenz95
from src.parsers import parser_generate_obs
from src.functions import generate_training_samples_ABC_model, LogLike

os.environ["QT_QPA_PLATFORM"] = 'offscreen'

parser = parser_generate_obs()
args = parser.parse_args()

model = args.model
start_observation_index = args.start_observation
n_observations = args.n_observations
sleep_time = args.sleep
results_folder = args.root_folder

default_root_folder = {"gaussian": "results/gaussian/",
                       "gamma": "results/gamma/",
                       "beta": "results/beta/",
                       "AR2": "results/AR2/",
                       "MA2": "results/MA2/",
                       "fullLorenz95": "results/fullLorenz95/",
                       "fullLorenz95smaller": "results/fullLorenz95smaller/"}
if results_folder is None:
    results_folder = default_root_folder[model]

observation_folder = results_folder + args.observation_folder + '/'

if sleep_time > 0:
    print("Wait for {} minutes...".format(sleep_time))
    sleep(60 * sleep_time)
    print("Done waiting!")

n_samples_true_MCMC = 20000
burnin_true_MCMC = 20000
cores = 2

seed = 1

save_true_MCMC_trace = True
save_observation = True

if model == "gaussian":
    mu_bounds = [-10, 10]
    sigma_bounds = [1, 10]
    theta_vect, samples_matrix = generate_gaussian_training_samples(n_theta=n_observations,
                                                                    size_iid_samples=10, seed=seed, mu_bounds=mu_bounds,
                                                                    sigma_bounds=sigma_bounds)
elif model == "beta":
    alpha_bounds = [1, 3]
    beta_bounds = [1, 3]
    theta_vect, samples_matrix = generate_beta_training_samples(n_theta=n_observations,
                                                                size_iid_samples=10, seed=seed,
                                                                alpha_bounds=alpha_bounds,
                                                                beta_bounds=beta_bounds)
elif model == "gamma":
    k_bounds = [1, 3]
    theta_bounds = [1, 3]
    theta_vect, samples_matrix = generate_gamma_training_samples(n_theta=n_observations,
                                                                 size_iid_samples=10, seed=seed, k_bounds=k_bounds,
                                                                 theta_bounds=theta_bounds)
elif model == "AR2":
    arma_size = 100
    ar1_bounds = [-1, 1]
    ar2_bounds = [-1, 0]

    ar1 = Uniform([[ar1_bounds[0]], [ar1_bounds[1]]], name='ar1')
    ar2 = Uniform([[ar2_bounds[0]], [ar2_bounds[1]]], name='ar2')
    arma_abc_model = ARMAmodel([ar1, ar2], num_AR_params=2, num_MA_params=0, size=arma_size)

    theta_vect, samples_matrix = generate_training_samples_ABC_model(arma_abc_model, n_observations, seed=seed)

elif model == "MA2":
    arma_size = 100
    ma1_bounds = [-1, 1]
    ma2_bounds = [0, 1]

    ma1 = Uniform([[ma1_bounds[0]], [ma1_bounds[1]]], name='ma1')
    ma2 = Uniform([[ma2_bounds[0]], [ma2_bounds[1]]], name='ma2')
    arma_abc_model = ARMAmodel([ma1, ma2], num_AR_params=0, num_MA_params=2, size=arma_size)

    theta_vect, samples_matrix = generate_training_samples_ABC_model(arma_abc_model, n_observations, seed=seed)

elif "Lorenz95" in model:

    theta1_min = 1.4
    theta1_max = 2.2
    theta2_min = 0
    theta2_max = 1

    sigma_e_min = 1.5
    sigma_e_max = 2.5
    phi_min = 0
    phi_max = 1

    theta1 = Uniform([[theta1_min], [theta1_max]], name='theta1')
    theta2 = Uniform([[theta2_min], [theta2_max]], name='theta2')
    sigma_e = Uniform([[sigma_e_min], [sigma_e_max]], name='sigma_e')
    phi = Uniform([[phi_min], [phi_max]], name='phi')

    lorenz = StochLorenz95([theta1, theta2, sigma_e, phi], time_units=1.5 if model == "fullLorenz95smaller" else 4,
                           n_timestep_per_time_unit=30, K=8 if model == "fullLorenz95smaller" else 40, name='lorenz', )
    theta_vect, samples_matrix = generate_training_samples_ABC_model(lorenz, n_observations, seed=seed)

for obs_index in range(start_observation_index, n_observations):
    print("Observation {}".format(obs_index + 1))
    if isinstance(samples_matrix, np.ndarray):
        x_obs = samples_matrix[obs_index]
    else:
        x_obs = samples_matrix[obs_index].numpy()
    if isinstance(theta_vect, np.ndarray):
        theta_obs = theta_vect[obs_index]
    else:
        theta_obs = theta_vect[obs_index].numpy()
    np.save(observation_folder + "theta_obs{}".format(obs_index + 1), theta_obs)
    np.save(observation_folder + "x_obs{}".format(obs_index + 1), x_obs)

    if model == "AR2":
        logl = LogLike(ar2_log_lik_for_mcmc, x_obs)
    elif model == "MA2":
        logl = LogLike(ma2_log_lik_for_mcmc, x_obs)

    if not "Lorenz95" in model:
        if model == "gaussian":
            with pm.Model() as model:
                mu = pm.Uniform('mu', mu_bounds[0], mu_bounds[1])
                sigma = pm.Uniform('sigma', sigma_bounds[0], sigma_bounds[1])
                # obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=x_obs)
                # on department desktop:
                obs = pm.Normal('obs', mu=mu, tau=1 / sigma ** 2, observed=x_obs)
                trace_true = pm.sample(n_samples_true_MCMC, tune=burnin_true_MCMC, cores=cores)
                trace_true = np.concatenate((trace_true['mu'].reshape(-1, 1), trace_true['sigma'].reshape(-1, 1)),
                                            axis=1)
        elif model == "beta":
            with pm.Model() as model:
                alpha = pm.Uniform('alpha', alpha_bounds[0], alpha_bounds[1])
                beta = pm.Uniform('beta', beta_bounds[0], beta_bounds[1])
                obs = pm.Beta('obs', alpha=alpha, beta=beta, observed=x_obs)
                trace_true = pm.sample(n_samples_true_MCMC, tune=burnin_true_MCMC, cores=cores)
                trace_true = np.concatenate((trace_true['alpha'].reshape(-1, 1), trace_true['beta'].reshape(-1, 1)),
                                            axis=1)
        elif model == "gamma":
            with pm.Model() as model:
                k = pm.Uniform('k', k_bounds[0], k_bounds[1])
                theta = pm.Uniform('theta', theta_bounds[0], theta_bounds[1])
                obs = pm.Gamma('obs', mu=k * theta, sigma=theta * np.sqrt(k), observed=x_obs)
                trace_true = pm.sample(n_samples_true_MCMC, tune=burnin_true_MCMC, cores=cores)
                trace_true = np.concatenate((trace_true['k'].reshape(-1, 1), trace_true['theta'].reshape(-1, 1)),
                                            axis=1)
        elif model == "AR2":
            with pm.Model() as model:
                ar1 = pm.Uniform('ar1', ar1_bounds[0], ar1_bounds[1], )
                ar2 = pm.Uniform('ar2', ar2_bounds[0], ar2_bounds[1])

                theta = tt.as_tensor_variable([ar1, ar2])

                pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
                trace_true = pm.sample(n_samples_true_MCMC, tune=burnin_true_MCMC, cores=cores)
                trace_true = np.concatenate((trace_true['ar1'].reshape(-1, 1), trace_true['ar2'].reshape(-1, 1)),
                                            axis=1)
        elif model == "MA2":
            with pm.Model() as model:
                ma1 = pm.Uniform('ma1', ma1_bounds[0], ma1_bounds[1], )
                ma2 = pm.Uniform('ma2', ma2_bounds[0], ma2_bounds[1])

                theta = tt.as_tensor_variable([ma1, ma2])

                pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})

                trace_true = pm.sample(n_samples_true_MCMC, tune=burnin_true_MCMC, cores=cores)
                trace_true = np.concatenate((trace_true['ma1'].reshape(-1, 1), trace_true['ma2'].reshape(-1, 1)),
                                            axis=1)

        if save_true_MCMC_trace:
            np.save(observation_folder + "true_mcmc_trace{}".format(obs_index + 1), trace_true)

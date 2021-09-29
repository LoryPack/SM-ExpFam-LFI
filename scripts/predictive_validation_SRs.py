import argparse
import os
import sys
from time import sleep

import numpy as np

sys.path.append(os.getcwd())

from src.functions import subsample_trace, DrawFromParamValues
from src.utils_Lorenz95_example import extract_posterior_mean_from_journal_Lorenz95, \
    extract_params_and_weights_from_journal_Lorenz95
from src.scoring_rules import estimate_kernel_score_timeseries, estimate_energy_score_timeseries, \
    estimate_bandwidth_timeseries

from abcpy.output import Journal
from abcpy.backends import BackendMPI, BackendDummy

from src.utils_Lorenz95_example import StochLorenz95

parser = argparse.ArgumentParser()
parser.add_argument('technique', type=str,
                    help="The technique to use; can be 'SM' or 'SSM' (both using exponential family),"
                         "or 'FP'. The latter does not work with exchange.")
parser.add_argument('model', type=str, help="The statistical model to consider.")
parser.add_argument('--inference_technique', type=str, default="exchange",
                    help="Inference approach; can be exchange or ABC.")
parser.add_argument('--sleep', type=float, default=0, help='Minutes to sleep before starting')
parser.add_argument('--start_observation_index', type=int, default=0, help='Index to start from')
parser.add_argument('--n_observations', type=int, default=100, help='Total number of observations.')
parser.add_argument('--n_samples', type=int, default=1000,
                    help='Number of samples for ABCpy journals (otherwise they do not load right)')
parser.add_argument('--subsample_size_exchange', type=int, default=1000,
                    help='Number of samples to take in the exchange MCMC results to generate the '
                         'predictive distribution')
parser.add_argument('--root_folder', type=str, default=None)
parser.add_argument('--observation_folder', type=str, default="observations")
parser.add_argument('--inference_folder', type=str, default="Exc-SM")
parser.add_argument('--ABC_alg', type=str, default="SABC",
                    help="ABC algorithm to use; can be PMCABC, APMCABC, SABC or RejectionABC")
parser.add_argument('--ABC_steps', type=int, default=3, help="Number of steps for sequential ABC algorithms.")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--load_results_if_available', action="store_true")
parser.add_argument('--use_MPI', '-m', action="store_true", help='Use MPI to generate simulations')
parser.add_argument('--gamma_kernel_score', type=float, default=None,
                    help='The value of bandwidth used in the kernel SR. If not provided, it is determined by running '
                         'simulations from the prior.')

args = parser.parse_args()

technique = args.technique
model = args.model
inference_technique = args.inference_technique
sleep_time = args.sleep
start_observation_index = args.start_observation_index
n_observations = args.n_observations
n_samples = args.n_samples
subsample_size_exchange = args.subsample_size_exchange
results_folder = args.root_folder
observation_folder = args.observation_folder
inference_folder = args.inference_folder
ABC_alg = args.ABC_alg
ABC_steps = args.ABC_steps
seed = args.seed
load_results_if_available = args.load_results_if_available
use_MPI = args.use_MPI
gamma_kernel_score = args.gamma_kernel_score

np.random.seed(seed)

# checks
if model not in ("Lorenz95", "fullLorenz95", "fullLorenz95smaller"):
    raise NotImplementedError

if inference_technique not in ("exchange", "ABC"):
    raise NotImplementedError

print(f"{model} model with {inference_technique} {technique}")
# set up the default root folder and other values
default_root_folder = {"Lorenz95": "results/Lorenz95/", "fullLorenz95": "results/fullLorenz95/",
                       "fullLorenz95smaller": "results/fullLorenz95smaller/"}
if results_folder is None:
    results_folder = default_root_folder[model]

results_folder = results_folder + '/'
observation_folder = results_folder + observation_folder + '/'
inference_folder = results_folder + inference_folder + '/'

if sleep_time > 0:
    print("Wait for {} minutes...".format(sleep_time))
    sleep(60 * sleep_time)
    print("Done waiting!")

# these values are not used really:
theta1 = 2
theta2 = 0.5
sigma_e = 1
phi = 0.4

time_units = 1.5 if model == "fullLorenz95smaller" else 4
num_vars_in_Lorenz = 8 if model == "fullLorenz95smaller" else 40
ABC_model = StochLorenz95([theta1, theta2, sigma_e, phi], time_units=time_units,
                          n_timestep_per_time_unit=30, name='lorenz', K=num_vars_in_Lorenz)
extract_posterior_mean_from_journal = extract_posterior_mean_from_journal_Lorenz95
extract_params_and_weights_from_journal = extract_params_and_weights_from_journal_Lorenz95

if inference_technique == "ABC":
    namefile_postfix_no_index = f"_{ABC_alg}_{technique}_steps_{ABC_steps}_n_samples_{n_samples}"
else:
    namefile_postfix_no_index = "_{}_n_samples_{}".format("exchange", subsample_size_exchange)

# attempt loading the results if required:
compute_srs = True
if load_results_if_available:
    try:
        energy_sr_values_timestep = np.load(
            inference_folder + "energy_sr_values_timestep" + namefile_postfix_no_index + ".npy")
        energy_sr_values_cumulative = np.load(
            inference_folder + "energy_sr_values_cumulative" + namefile_postfix_no_index + ".npy")
        kernel_sr_values_timestep = np.load(
            inference_folder + "kernel_sr_values_timestep" + namefile_postfix_no_index + ".npy")
        kernel_sr_values_cumulative = np.load(
            inference_folder + "kernel_sr_values_cumulative" + namefile_postfix_no_index + ".npy")
        print("Loaded previously computed scoring rule values.")
        compute_srs = False
    except FileNotFoundError:
        pass

if compute_srs:  # compute_srs:
    backend = BackendMPI() if use_MPI else BackendDummy()
    if gamma_kernel_score is None:
        print("Set gamma from simulations from the model")
        gamma_kernel_score = estimate_bandwidth_timeseries(ABC_model, backend=backend, n_theta=1000, seed=seed + 1,
                                                           num_vars=num_vars_in_Lorenz)
        print("Estimated gamma ", gamma_kernel_score)

    print("Computing scoring rules values by generating predictive distribution.")

    draw_from_params = DrawFromParamValues([ABC_model], backend=backend, seed=seed)

    energy_sr_values_cumulative = np.empty(n_observations - start_observation_index)
    energy_sr_values_timestep = np.empty((n_observations - start_observation_index, int(time_units * 30)))
    kernel_sr_values_cumulative = np.empty(n_observations - start_observation_index)
    kernel_sr_values_timestep = np.empty((n_observations - start_observation_index, int(time_units * 30)))

    for obs_index in range(start_observation_index, n_observations):
        print("Observation ", obs_index + 1)
        namefile_postfix = "_{}".format(obs_index + 1) + namefile_postfix_no_index

        # load the actual observation
        if "fullLorenz95" in model:
            x_obs = np.load(observation_folder + "x_obs{}.npy".format(obs_index + 1))
        else:
            x_obs = np.load(observation_folder + "timeseriers_obs{}.npy".format(obs_index + 1))
        # theta_obs = np.load(observation_folder + "theta_obs{}.npy".format(obs_index + 1))
        # reshape the observation:
        x_obs = x_obs.reshape(num_vars_in_Lorenz, -1)
        # print(x_obs.shape)

        # load now the posterior for that observation
        if inference_technique == "ABC":
            jrnl_ABC = Journal.fromFile(inference_folder + "jrnl" + namefile_postfix + ".jnl")
            params, weights = extract_params_and_weights_from_journal(jrnl_ABC)
            # subsample journal according to weights (bootstrap):
            # params_ABC_subsampled = subsample_params_according_to_weights(params_ABC, weights_ABC,
            #                                                                  size=n_post_samples)
        else:
            trace_exchange = np.load(inference_folder + f"exchange_mcmc_trace{obs_index + 1}.npy")
            # subsample trace:
            params = subsample_trace(trace_exchange, size=subsample_size_exchange)
            weights = None

        # print("Results loaded correctly")

        # now simulate for all the different param values
        # print("Simulate...")
        n_posterior_samples = n_samples if inference_technique == "ABC" else subsample_size_exchange
        posterior_simulations_params, posterior_simulations = draw_from_params.sample(params)
        # print("Done!")
        # print(posterior_simulations.shape)
        posterior_simulations = posterior_simulations.reshape(n_posterior_samples, num_vars_in_Lorenz,
                                                              -1)  # last index is the timestep
        # print(posterior_simulations.shape)

        # estimate the SR for that observation and cumulate over the timesteps
        energy_scores = estimate_energy_score_timeseries(posterior_simulations, x_obs)
        energy_sr_values_timestep[obs_index - start_observation_index] = energy_scores[0]
        energy_sr_values_cumulative[obs_index - start_observation_index] = energy_scores[1]

        kernel_scores = estimate_kernel_score_timeseries(posterior_simulations, x_obs, sigma=gamma_kernel_score)
        kernel_sr_values_timestep[obs_index - start_observation_index] = kernel_scores[0]
        kernel_sr_values_cumulative[obs_index - start_observation_index] = kernel_scores[1]

    np.save(inference_folder + "energy_sr_values_timestep" + namefile_postfix_no_index, energy_sr_values_timestep)
    np.save(inference_folder + "energy_sr_values_cumulative" + namefile_postfix_no_index, energy_sr_values_cumulative)
    np.save(inference_folder + "kernel_sr_values_timestep" + namefile_postfix_no_index, kernel_sr_values_timestep)
    np.save(inference_folder + "kernel_sr_values_cumulative" + namefile_postfix_no_index, kernel_sr_values_cumulative)

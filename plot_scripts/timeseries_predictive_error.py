import argparse
import os
import sys
from time import sleep

import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())

from src.functions import plot_confidence_bands_performance_vs_iteration, subsample_trace
from src.utils_Lorenz95_example import extract_posterior_mean_from_journal_Lorenz95, \
    extract_params_and_weights_from_journal_Lorenz95

from abcpy.output import Journal

from src.utils_Lorenz95_example import StochLorenz95

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help="The statistical model to consider.")
parser.add_argument('compare_with', type=str, help="With which approach to compare ABC_FP; can be "
                                                   "Exc-SM, Exc-SSM, ABC-SM or ABC-SSM. ")
parser.add_argument('--sleep', type=float, default=0, help='Minutes to sleep before starting')
parser.add_argument('--start_observation_index', type=int, default=0, help='Index to start from')
parser.add_argument('--n_observations', type=int, default=100, help='Total number of observations.')
parser.add_argument('--root_folder', type=str, default=None)
parser.add_argument('--observation_folder', type=str, default="observations")
parser.add_argument('--exchange_folder', type=str, default="inferences_ssm")
parser.add_argument('--ABC_FP_folder', type=str, default="ABC-FP")
parser.add_argument('--ABC_FP_algorithm', type=str, default="SABC")
parser.add_argument('--ABC_FP_n_samples', type=int, default=1000)
parser.add_argument('--ABC_FP_steps', type=int, default=100)
parser.add_argument('--ABC_SM_folder', type=str, default="ABC-SM")
parser.add_argument('--ABC_SM_algorithm', type=str, default="SABC")
parser.add_argument('--ABC_SM_n_samples', type=int, default=1000)
parser.add_argument('--ABC_SM_steps', type=int, default=100)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--load_errors_if_available', action="store_true")
parser.add_argument('--RMSE_only', action="store_true")

args = parser.parse_args()

model = args.model
compare_with = args.compare_with
sleep_time = args.sleep
start_observation_index = args.start_observation_index
n_observations = args.n_observations
results_folder = args.root_folder
observation_folder = args.observation_folder
exchange_folder = args.exchange_folder
ABC_FP_folder = args.ABC_FP_folder
ABC_FP_algorithm = args.ABC_FP_algorithm
ABC_FP_steps = args.ABC_FP_steps
ABC_FP_n_samples = args.ABC_FP_n_samples
ABC_SM_folder = args.ABC_SM_folder
ABC_SM_algorithm = args.ABC_SM_algorithm
ABC_SM_steps = args.ABC_SM_steps
ABC_SM_n_samples = args.ABC_SM_n_samples
seed = args.seed
load_errors_if_available = args.load_errors_if_available
RMSE_only = args.RMSE_only

time_units_future = 2
n_timestep_per_time_unit = 30

np.random.seed(seed)
rng = np.random.RandomState(seed)

# checks
if model not in ("Lorenz95", "fullLorenz95", "fullLorenz95smaller"):
    raise NotImplementedError

print("{} model.".format(model))
# set up the default root folder and other values
default_root_folder = {"Lorenz95": "results/Lorenz95/", "fullLorenz95": "results/fullLorenz95/",
                       "fullLorenz95": "results/fullLorenz95smaller/"}
if results_folder is None:
    results_folder = default_root_folder[model]
if compare_with not in ("Exc-SM", "Exc-SSM", "ABC-SM", "ABC-SSM"):
    raise RuntimeError

results_folder = results_folder + '/'
observation_folder = results_folder + observation_folder + '/'
exchange_folder = results_folder + exchange_folder + '/'
ABC_FP_folder = results_folder + ABC_FP_folder + '/'
ABC_SM_folder = results_folder + ABC_SM_folder + '/'
my_technique_folder = exchange_folder if "exc" in compare_with else ABC_SM_folder

if sleep_time > 0:
    print("Wait for {} minutes...".format(sleep_time))
    sleep(60 * sleep_time)
    print("Done waiting!")

if "Lorenz95" in model:
    # these values are not used really:
    theta1 = 2
    theta2 = 0.5
    sigma_e = 1
    phi = 0.4

    ABC_model = StochLorenz95([theta1, theta2, sigma_e, phi], time_units=time_units_future,
                              n_timestep_per_time_unit=n_timestep_per_time_unit, name='lorenz',
                              K=8 if model == "fullLorenz95smaller" else 40)
    extract_posterior_mean_from_journal = extract_posterior_mean_from_journal_Lorenz95
    extract_params_and_weights_from_journal = extract_params_and_weights_from_journal_Lorenz95

ABC_FP_namefile_postfix_no_index = f"_{ABC_FP_algorithm}_FP_steps_{ABC_FP_steps}_n_samples_{ABC_FP_n_samples}"
ABC_SM_namefile_postfix_no_index = f"_{ABC_SM_algorithm}_{'SSM' if 'SSM' in compare_with else 'SM'}_steps_{ABC_SM_steps}_n_samples_{ABC_SM_n_samples}"

# here instead is with posterior expectation of errors.
#  E[||y_0 - y(theta)||_2^2]
n_post_samples_exchange = 1000

compute_errors = True
if load_errors_if_available:
    try:
        relative_errors_post_exp = np.load(my_technique_folder + "relative_errors_post_exp_wrt_ABC_FP.npy")
        print("Loaded previously computed relative errors with posterior expectation.")
        compute_errors = False
    except FileNotFoundError:
        pass

if compute_errors:
    print("Computing relative errors with posterior expectation.")
    relative_errors_post_exp = np.empty(
        (n_observations - start_observation_index, time_units_future * n_timestep_per_time_unit))

    for obs_index in range(start_observation_index, n_observations):
        print("Observation ", obs_index + 1)
        ABC_FP_namefile_postfix = f"_{obs_index + 1}" + ABC_FP_namefile_postfix_no_index
        ABC_SM_namefile_postfix = f"_{obs_index + 1}" + ABC_SM_namefile_postfix_no_index
        # namefile_postfix = "_{}".format(obs_index + 1) + namefile_postfix_no_index
        if "fullLorenz95" in model:
            x_obs = np.load(observation_folder + "x_obs{}.npy".format(obs_index + 1))
        else:
            x_obs = np.load(observation_folder + "timeseriers_obs{}.npy".format(obs_index + 1))
        theta_obs = np.load(observation_folder + "theta_obs{}.npy".format(obs_index + 1))

        # need to load now 1) my own method and 2) the ABC_FP method
        if "exc" in compare_with:
            trace_exchange = np.load(exchange_folder + f"exchange_mcmc_trace{obs_index + 1}.npy")
            # subsample trace:
            trace_exchange_subsampled = subsample_trace(trace_exchange, size=n_post_samples_exchange)
            predictive_error_exchange = np.zeros(  # for 1)
                (n_post_samples_exchange, time_units_future * n_timestep_per_time_unit))
            future_evolution_exchange_array = np.zeros(  # for 2)
                (n_post_samples_exchange, 40, time_units_future * n_timestep_per_time_unit))
        else:
            jrnl_ABC_SM = Journal.fromFile(ABC_SM_folder + "jrnl" + ABC_SM_namefile_postfix + ".jnl")
            # subsample journal according to weights (bootstrap):
            params_ABC_SM, weights_ABC_SM = extract_params_and_weights_from_journal(jrnl_ABC_SM)
            # params_ABC_SM_subsampled = subsample_params_according_to_weights(params_ABC_SM, weights_ABC_SM,
            #                                                                  size=n_post_samples)
            predictive_error_ABC_SM = np.zeros((len(params_ABC_SM), time_units_future * n_timestep_per_time_unit))  # 1)
            future_evolution_ABC_SM_array = np.zeros(  # for 2)
                (len(params_ABC_SM), 40, time_units_future * n_timestep_per_time_unit))

        jrnl_ABC_FP = Journal.fromFile(ABC_FP_folder + "jrnl" + ABC_FP_namefile_postfix + ".jnl")
        # subsample journal according to weights (bootstrap):
        params_ABC_FP, weights_ABC_FP = extract_params_and_weights_from_journal(jrnl_ABC_FP)
        # params_ABC_FP_subsampled = subsample_params_according_to_weights(params_ABC_FP, weights_ABC_FP,
        #                                                                  size=n_post_samples)
        predictive_error_ABC_FP = np.zeros((len(params_ABC_FP), time_units_future * n_timestep_per_time_unit))  # 1)
        future_evolution_ABC_FP_array = np.zeros(  # for 2)
            (len(params_ABC_FP), 40, time_units_future * n_timestep_per_time_unit))

        # this is same for all post samples
        # set dynamical model to initial value which is the end of the observation window
        ABC_model._set_initial_state(initial_state=x_obs.reshape(40, -1)[:, -1])
        future_evolution_true_param = ABC_model.forward_simulate(theta_obs, k=1, rng=rng)[0].reshape(40, -1)
        # loops over the post samples:
        if "exc" in compare_with:
            for post_sample in range(n_post_samples_exchange):
                # evolve the model with all different posterior samples
                future_evolution_exchange = \
                    ABC_model.forward_simulate(trace_exchange_subsampled[post_sample], k=1, rng=rng)[0].reshape(40, -1)
                predictive_error_exchange[post_sample] = np.linalg.norm(  # 1)
                    future_evolution_true_param - future_evolution_exchange, axis=0)
                future_evolution_exchange_array[post_sample] = future_evolution_exchange  # 2)
        else:
            for post_sample in range(len(params_ABC_SM)):
                future_evolution_ABC_SM = \
                    ABC_model.forward_simulate(params_ABC_SM[post_sample], k=1, rng=rng)[
                        0].reshape(40, -1)
                predictive_error_ABC_SM[post_sample] = np.linalg.norm(  # 1)
                    future_evolution_true_param - future_evolution_ABC_SM, axis=0)
                future_evolution_ABC_SM_array[post_sample] = future_evolution_ABC_SM  # 2)
                # check norms
        for post_sample in range(len(params_ABC_FP)):
            future_evolution_ABC_FP = ABC_model.forward_simulate(params_ABC_FP[post_sample], k=1, rng=rng)[
                0].reshape(40, -1)
            predictive_error_ABC_FP[post_sample] = np.linalg.norm(future_evolution_true_param - future_evolution_ABC_FP,
                                                                  axis=0)  # 1)
            future_evolution_ABC_FP_array[post_sample] = future_evolution_ABC_FP  # 2)

        # compute now the expectation over the post samples:
        weights_ABC_FP /= np.sum(weights_ABC_FP)  # normalize weights
        mean_predictive_error_ABC_FP = np.einsum('i,ij->j', weights_ABC_FP, predictive_error_ABC_FP)  # 1)
        mean_future_evolution_ABC_FP = np.einsum('i,ijk->jk', weights_ABC_FP, future_evolution_ABC_FP_array)  # 2)
        predictive_error_mean_future_evolution_ABC_FP = np.linalg.norm(
            future_evolution_true_param - mean_future_evolution_ABC_FP, axis=0)  # 2)
        if "exc" in compare_with:
            mean_predictive_error_exchange = np.mean(predictive_error_exchange, axis=0)  # 1)
            mean_future_evolution_exchange = np.mean(future_evolution_exchange_array, axis=0)  # 2)
            predictive_error_mean_future_evolution_exchange = np.linalg.norm(
                future_evolution_true_param - mean_future_evolution_exchange, axis=0)  # 2)
        else:
            weights_ABC_SM /= np.sum(weights_ABC_SM)  # normalize weights
            mean_predictive_error_ABC_SM = np.einsum('i,ij->j', weights_ABC_SM, predictive_error_ABC_SM)  # 1)
            mean_future_evolution_ABC_SM = np.einsum('i,ijk->jk', weights_ABC_SM, future_evolution_ABC_SM_array)  # 2)
            predictive_error_mean_future_evolution_ABC_SM = np.linalg.norm(
                future_evolution_true_param - mean_future_evolution_ABC_SM, axis=0)  # 2)

        # compute the relative errors:
        if "exc" in compare_with:
            relative_errors_post_exp[obs_index - start_observation_index] = \
                (mean_predictive_error_ABC_FP - mean_predictive_error_exchange) / mean_predictive_error_ABC_FP  # 1)
        else:
            relative_errors_post_exp[obs_index - start_observation_index] = \
                (mean_predictive_error_ABC_FP - mean_predictive_error_ABC_SM) / mean_predictive_error_ABC_FP  # 1)

    np.save(my_technique_folder + "relative_errors_post_exp_wrt_ABC_FP", relative_errors_post_exp)

fig, ax = plot_confidence_bands_performance_vs_iteration(relative_errors_post_exp.transpose(1, 0), start_step=0)
ax.axhline(0, color="blue")
ax.set_title(compare_with)
ax.set_xlabel(r"$t$")
ax.set_xticks([0, 14, 29, 44, 59])
ax.set_xticklabels([4, 4.5, 5, 5.5, 6])
ax.set_ylabel(r"Relative decrease in prediction error $\zeta(t)$")
plt.savefig(my_technique_folder + "relative_errors_post_exp_wrt_ABC_FP.pdf", bbox_inches="tight")
plt.close()
# plt.show()

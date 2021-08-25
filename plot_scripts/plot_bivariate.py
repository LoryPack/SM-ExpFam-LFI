import argparse
import os
import sys
from time import sleep

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox

sys.path.append(os.getcwd())
from src.functions import plot_bivariate_marginal_with_trace_samples

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help="The statistical model to consider.")
parser.add_argument('--sleep', type=float, default=0, help='Minutes to sleep before starting')
parser.add_argument('--start_observation_index', type=int, default=0, help='Index to start from')
parser.add_argument('--n_observations', type=int, default=100, help='Total number of observations.')
parser.add_argument('--n_samples', type=int, default=10000, help='Number of samples (for ABC or exchange MCMC)')
parser.add_argument('--burnin_MCMC', type=int, default=10000, help='Burnin samples for exchange MCMC.')
parser.add_argument('--root_folder', type=str, default=None)
parser.add_argument('--observation_folder', type=str, default="observations")
parser.add_argument('--inference_folder', type=str, default="Exc-SM")
parser.add_argument('--subsample_size', type=int, default=1000,
                    help='Number of samples used for bivariate plots (if required).')

args = parser.parse_args()

model = args.model
sleep_time = args.sleep
start_observation_index = args.start_observation_index
n_observations = args.n_observations
n_samples = args.n_samples
burnin_MCMC = args.burnin_MCMC
results_folder = args.root_folder
observation_folder = args.observation_folder
inference_folder = args.inference_folder
subsample_size = args.subsample_size

# checks
if model not in ("gaussian", "beta", "gamma", "MA2", "AR2"):
    raise NotImplementedError

print("{} model.".format(model))
# set up the default root folder and other values
default_root_folder = {"gaussian": "results/gaussian/",
                       "gamma": "results/gamma/",
                       "beta": "results/beta/",
                       "AR2": "results/AR2/",
                       "MA2": "results/MA2/"}
if results_folder is None:
    results_folder = default_root_folder[model]

results_folder = results_folder + '/'
observation_folder = results_folder + observation_folder + '/'
inference_folder = results_folder + inference_folder + '/'

if sleep_time > 0:
    print("Wait for {} minutes...".format(sleep_time))
    sleep(60 * sleep_time)
    print("Done waiting!")

plot_boxplots = False  # maybe add this back! In a postprocessing script!
save_exchange_MCMC_trace = True
save_jrnl = True

namefile_postfix_no_index = "_{}_n_samples_{}".format("exchange", n_samples)

if model == "gaussian":
    mu_bounds = [-10, 10]
    sigma_bounds = [1, 10]
    lower_bounds = np.array([mu_bounds[0], sigma_bounds[0]])
    upper_bounds = np.array([mu_bounds[1], sigma_bounds[1]])
    param_names = [r"$\mu$", r"$\sigma$"]
elif model == "beta":
    alpha_bounds = [1, 3]
    beta_bounds = [1, 3]
    lower_bounds = np.array([alpha_bounds[0], beta_bounds[0]])
    upper_bounds = np.array([alpha_bounds[1], beta_bounds[1]])
    param_names = [r"$\alpha$", r"$\beta$"]
elif model == "gamma":
    k_bounds = [1, 3]
    theta_bounds = [1, 3]
    lower_bounds = np.array([k_bounds[0], theta_bounds[0]])
    upper_bounds = np.array([k_bounds[1], theta_bounds[1]])
    param_names = [r"$k$", r"$\theta$"]
elif model == "AR2":
    arma_size = 100
    ar1_bounds = [-1, 1]
    ar2_bounds = [-1, 0]
    lower_bounds = np.array([ar1_bounds[0], ar2_bounds[0]])
    upper_bounds = np.array([ar1_bounds[1], ar2_bounds[1]])
    param_names = [r"$\theta_1$", r"$\theta_2$"]
elif model == "MA2":
    arma_size = 100
    ma1_bounds = [-1, 1]
    ma2_bounds = [0, 1]
    lower_bounds = np.array([ma1_bounds[0], ma2_bounds[0]])
    upper_bounds = np.array([ma1_bounds[1], ma2_bounds[1]])
    param_names = [r"$\theta_1$", r"$\theta_2$"]

for obs_index in range(start_observation_index, n_observations):
    print("Observation ", obs_index + 1)
    namefile_postfix = "_{}".format(obs_index + 1) + namefile_postfix_no_index
    x_obs = np.load(observation_folder + "x_obs{}.npy".format(obs_index + 1))
    theta_obs = np.load(observation_folder + "theta_obs{}.npy".format(obs_index + 1))
    trace_true = np.load(observation_folder + "true_mcmc_trace{}.npy".format(obs_index + 1))

    trace_exchange = np.load(inference_folder + "exchange_mcmc_trace{}.npy".format(obs_index + 1))

    fig, ax = plot_bivariate_marginal_with_trace_samples(
        theta_obs, trace_exchange, trace_true,
        thetarange=np.array([lower_bounds, upper_bounds]),
        namefile=None, color="#40739E",
        figsize_vertical=5, title_size=16, label_size=16, param1_name=param_names[0],
        param2_name=param_names[1], space_subplots=0.3, vertical=True)

    namefile = inference_folder + "bivariate_marginals" + namefile_postfix + ".pdf"
    bbox_inches = Bbox(np.array([[-0.25, 0.5], [4.7, 9.1]]))
    plt.savefig(namefile, bbox_inches=bbox_inches)

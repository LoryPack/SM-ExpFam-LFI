import argparse

import matplotlib.pyplot as plt
import numpy as np

from abcpy.output import Journal

parser = argparse.ArgumentParser()
parser.add_argument('technique', type=str, help="The technique to use; can be 'SM' (exponential family) or 'FP'.")
parser.add_argument('--inference_technique', type=str, default="exchange",
                    help="Inference approach; can be exchange or ABC.")
parser.add_argument('--root_folder', type=str, default=None)
parser.add_argument('--observation_folder', type=str, default="observations")
parser.add_argument('--inference_folder', type=str)
parser.add_argument('--ABC_alg', type=str, default="SABC",
                    help="ABC algorithm to use; can be PMCABC, APMCABC, SABC or RejectionABC")
parser.add_argument('--ABC_steps', type=int, default=100, help="Number of steps for sequential ABC algorithms.")
parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples (for ABC or exchange MCMC)')
parser.add_argument('--obs_index', type=int, default=13)

args = parser.parse_args()

technique = args.technique
inference_technique = args.inference_technique
results_folder = args.root_folder
observation_folder = args.observation_folder
inference_folder = args.inference_folder
ABC_algorithm = args.ABC_alg
ABC_steps = args.ABC_steps
n_samples = args.n_samples
obs_index = args.obs_index

if results_folder is None:
    results_folder = "results/Lorenz95/"
# the one in the two results folders should be the same.
observation_folder = results_folder + args.observation_folder + "/"
inference_folder = results_folder + args.inference_folder + "/"

if inference_technique == "ABC":
    namefile_postfix_no_index = "_{}_{}_steps_{}_n_samples_{}".format(ABC_algorithm, technique, ABC_steps, n_samples)
elif inference_technique == "exchange":
    namefile_postfix_no_index = "_{}_n_samples_{}".format("exchange", n_samples)

namefile_postfix = "_{}".format(obs_index + 1) + namefile_postfix_no_index
x_obs = np.load(observation_folder + "x_obs{}.npy".format(obs_index + 1))
theta_obs = np.load(observation_folder + "theta_obs{}.npy".format(obs_index + 1))

if inference_technique == "ABC":
    jrnl = Journal.fromFile(inference_folder + "jrnl" + namefile_postfix + ".jnl")

else:
    trace_exchange_burned_in = np.load(inference_folder + "exchange_mcmc_trace{}.npy".format(obs_index + 1))
    parameters = [("theta1", trace_exchange_burned_in[:, 0].reshape(-1, 1, 1)),
                  ("theta2", trace_exchange_burned_in[:, 1].reshape(-1, 1, 1)),
                  ("sigma_e", trace_exchange_burned_in[:, 2].reshape(-1, 1, 1)),
                  ("phi", trace_exchange_burned_in[:, 3].reshape(-1, 1, 1)), ]

    # do the plot for exchange result as well by storing parameters in a journal:
    jrnl = Journal(0)
    jrnl.add_user_parameters(parameters)
    jrnl.add_weights(np.ones((trace_exchange_burned_in.shape[0], 1)))

param_names = [r"$\theta_1$", r"$\theta_2$", r"$\sigma_e$", r"$\phi$"]
theta1_min = 1.4
theta1_max = 2.2
theta2_min = 0
theta2_max = 1
sigma_e_min = 1.5
sigma_e_max = 2.5
phi_min = 0
phi_max = 1

ranges = dict([("theta1", [theta1_min, theta1_max]), ("theta2", [theta2_min, theta2_max]),
          ("sigma_e", [sigma_e_min, sigma_e_max]), ("phi", [phi_min, phi_max]), ])

fig, axes = jrnl.plot_posterior_distr(true_parameter_values=theta_obs, show_samples=False, show_density_values=False,
                                      figsize=10, contour_levels=14, write_posterior_mean=False, ranges_parameters=ranges)

figsize_actual = 4 * len(param_names)
figsize_actual = 28
label_size = figsize_actual / len(param_names) * 4
title_size = figsize_actual / len(param_names) * 4.25

for j, label in enumerate(param_names):
    axes[0, j].set_title(label, size=title_size)

    axes[j, 0].set_ylabel(label, size=label_size)
    axes[-1, j].set_xlabel(label, size=label_size)

plt.savefig(inference_folder + "joint_posterior" + namefile_postfix + ".pdf", bbox_inches="tight")

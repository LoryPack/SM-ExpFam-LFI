import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

sys.path.append(os.getcwd())

from src.utils_arma_example import extract_params_and_weights_from_journal_ar2, \
    extract_params_and_weights_from_journal_ma2, extract_posterior_mean_from_journal_ar2, \
    extract_posterior_mean_from_journal_ma2
from src.utils_beta_example import extract_params_and_weights_from_journal_beta, \
    extract_posterior_mean_from_journal_beta
from src.utils_gaussian_example import extract_params_and_weights_from_journal_gaussian, \
    extract_posterior_mean_from_journal_gaussian
from src.utils_gamma_example import extract_params_and_weights_from_journal_gamma, \
    extract_posterior_mean_from_journal_gamma
from src.functions import try_loading_wass_RMSE_post_mean

parser = argparse.ArgumentParser()
parser.add_argument('model',
                    help="The statistical model to consider; can be 'AR2', 'MA2', 'beta', 'gamma', 'gaussian'")
parser.add_argument('--observation_folder', type=str, default="observations")
parser.add_argument('--inferences_folder', type=str, default="Exc-SM")
parser.add_argument('--root_folder', type=str, default=None)
parser.add_argument('--n_samples', type=int, default=10000, help='Number of exchangeMCMC samples')
parser.add_argument('--CI_level', type=float, default=95,
                    help="The size of confidence interval (CI) to produce the plots. It represents the confidence "
                         "intervals in the plots vs iterations, and the position of the whiskers in the boxplots.")
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

model = args.model
results_folder = args.root_folder
CI_level = args.CI_level
seed = args.seed
n_samples = args.n_samples

np.random.seed(seed)  # seed rng
default_root_folder = {"gaussian": "results/gaussian/",
                       "gamma": "results/gamma/",
                       "beta": "results/beta/",
                       "AR2": "results/AR2/",
                       "MA2": "results/MA2/"}

extract_params_and_weights_fcns = {"gaussian": extract_params_and_weights_from_journal_gaussian,
                                   "gamma": extract_params_and_weights_from_journal_gamma,
                                   "beta": extract_params_and_weights_from_journal_beta,
                                   "AR2": extract_params_and_weights_from_journal_ar2,
                                   "MA2": extract_params_and_weights_from_journal_ma2}

extract_posterior_mean_fcns = {"gaussian": extract_posterior_mean_from_journal_gaussian,
                               "gamma": extract_posterior_mean_from_journal_gamma,
                               "beta": extract_posterior_mean_from_journal_beta,
                               "AR2": extract_posterior_mean_from_journal_ar2,
                               "MA2": extract_posterior_mean_from_journal_ma2}
model_text = {"gaussian": "Gaussian", "gamma": "Gamma", "beta": "Beta", "AR2": "AR(2)", "MA2": "MA(2)"}

if results_folder is None:
    results_folder = default_root_folder[model]
# the one in the two results folders should be the same.
observation_folder = results_folder + args.observation_folder + "/"
inferences_folder = results_folder + args.inferences_folder + "/"

extract_params_and_weights_from_journal = extract_params_and_weights_fcns[model]
extract_posterior_mean_from_journal = extract_posterior_mean_fcns[model]

if model not in ("gaussian", "beta", "gamma", "MA2", "AR2"):
    raise NotImplementedError

true_posterior_available = model in ("gaussian", "beta", "gamma", "MA2", "AR2")

true_model_inner_MCMC_only = False

print(f"{model} model.")

list_inner_MCMC_steps = [10, 30, 100, 200]

list_names = []
list_names_short = []
list_wass_dist = []
list_RMSE_post_mean = []

# load Wass dist and RMSE; assume they have already been computed:

if not true_model_inner_MCMC_only:
    for inner_MCMC_steps in list_inner_MCMC_steps:
        inference_folder_exchange_SM = inferences_folder + f"/{inner_MCMC_steps}_inn_steps/"
        # load wass dist and RMSE from the correct folder:
        wass_dist_exchange_SM, RMSE_post_mean_exchange_SM, load_successful_exchange_SM = try_loading_wass_RMSE_post_mean(
            inference_folder_exchange_SM, f"_exchange_n_samples_{n_samples}", True)
        if load_successful_exchange_SM:
            list_names.append(f"Observation {inner_MCMC_steps}")
            list_names_short.append(f"{inner_MCMC_steps}")
            list_wass_dist.append(wass_dist_exchange_SM)
            list_RMSE_post_mean.append(RMSE_post_mean_exchange_SM)

# wass dist and RMSE:

list_indeces = (np.arange(len(list_names)) + 1).tolist()

fig = plt.figure(figsize=(2 * len(list_names), 6))
# boxplots: the box covers range from 1st to 3rd quartile. We set the whiskers to the same band size in the
# confidence intervals plots
plt.boxplot(list_wass_dist, whis=(50 - CI_level / 2, 50 + CI_level / 2))
plt.xticks(list_indeces, list_names)
plt.ylabel("Wasserstein distance")
# plt.show()
plt.savefig(results_folder + "wass_dist_different_inner_MCMC_steps.png")
plt.close()

fig = plt.figure(figsize=(2 * len(list_names), 6))
plt.boxplot(list_RMSE_post_mean, whis=(50 - CI_level / 2, 50 + CI_level / 2))
plt.xticks(list_indeces, list_names)
plt.ylabel("RMSE post mean")
# plt.show()
plt.savefig(results_folder + "RMSE_post_mean_different_inner_MCMC_steps.png")
plt.close()

fontsize = 32
ticksize = 20
# joint plot:
fig, ax1 = plt.subplots(1, figsize=(0.8 * len(list_names) + 1, 8))
ax2 = ax1.twinx()
ax1.set_title(f"{model_text[model]}", fontsize=fontsize)
c = "blue"
ax1.boxplot(list_wass_dist, whis=(50 - CI_level / 2, 50 + CI_level / 2), patch_artist=True, notch=True,
            boxprops=dict(facecolor=c, color=c, alpha=0.5),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=c), widths=0.3, positions=np.array(list_indeces) - 0.15)
ax1.set_xticks(list_indeces)
ax1.tick_params(axis='y', labelcolor=c)
ax1.set_xticklabels(list_names_short, rotation=0)
ax1.tick_params(axis='both', which='major', labelsize=ticksize)
ax1.set_xlabel("Inner MCMC steps", fontsize=fontsize)
ax1.set_ylabel("Wasserstein distance", color=c, fontsize=fontsize)
ax1.set_ylim(ymin=0)
c = "red"
ax2.boxplot(list_RMSE_post_mean, whis=(50 - CI_level / 2, 50 + CI_level / 2), patch_artist=True, notch=True,
            boxprops=dict(facecolor=c, color=c, alpha=0.5),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=c), widths=0.3, positions=np.array(list_indeces) + 0.15)
ax2.set_xticks(list_indeces)
ax2.tick_params(axis='y', labelcolor=c)
ax2.set_xticklabels(list_names_short, rotation=0)
ax2.tick_params(axis='both', which='major', labelsize=ticksize)
ax2.set_xlabel("Inner MCMC steps", fontsize=fontsize)
ax2.set_ylabel("RMSE post mean", color=c, fontsize=fontsize)
ax2.set_ylim(ymin=0)

bbox_inches = Bbox(np.array([[-0.53, 0], [0.8 * len(list_names) + 1.8, 7.6]]))
plt.savefig(results_folder + "joint_boxplot_different_inner_MCMC_steps.pdf", bbox_inches=bbox_inches)
plt.close()

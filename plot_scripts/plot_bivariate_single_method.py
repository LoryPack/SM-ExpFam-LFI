import argparse
import os
import sys
from time import sleep

import numpy as np
from abcpy.output import Journal
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox

sys.path.append(os.getcwd())
from src.functions import plot_bivariate_marginal_with_trace_samples, subsample_trace
from src.utils_arma_example import extract_params_and_weights_from_journal_ar2, \
    extract_params_and_weights_from_journal_ma2
from src.utils_beta_example import extract_params_and_weights_from_journal_beta
from src.utils_gaussian_example import extract_params_and_weights_from_journal_gaussian
from src.utils_gamma_example import extract_params_and_weights_from_journal_gamma

parser = argparse.ArgumentParser()
parser.add_argument('technique', type=str,
                    help="The technique to use; can be 'SM' or 'SSM' (both using exponential family),"
                         "'FP' or 'True'. 'FP' does not work with exchange, and 'True' ignores the "
                         "'--inference_technique' argument and uses the true posterior distribution.")
parser.add_argument('model', type=str, help="The statistical model to consider.")
parser.add_argument('--inference_technique', type=str, default="exchange",
                    help="Inference approach; can be exchange or ABC.")
parser.add_argument('--sleep', type=float, default=0, help='Minutes to sleep before starting')
parser.add_argument('--start_observation_index', type=int, default=0, help='Index to start from')
parser.add_argument('--n_observations', type=int, default=100, help='Total number of observations.')
parser.add_argument('--n_samples', type=int, default=10000, help='Number of samples (for ABC or exchange MCMC)')
parser.add_argument('--burnin_MCMC', type=int, default=10000, help='Burnin samples for exchange MCMC.')
parser.add_argument('--ABC_alg', type=str, default="SABC",
                    help="ABC algorithm to use; can be PMCABC, APMCABC, SABC or RejectionABC")
parser.add_argument('--ABC_steps', type=int, default=100, help="Number of steps for sequential ABC algorithms.")
parser.add_argument('--root_folder', type=str, default=None)
parser.add_argument('--observation_folder', type=str, default="observations")
parser.add_argument('--inference_folder', type=str, default="Exc-SM")
parser.add_argument('--subsample_size', type=int, default=1000,
                    help='Number of samples used for bivariate plots (if required).')

args = parser.parse_args()

technique = args.technique
model = args.model
inference_technique = args.inference_technique  # this defaults to exchange
sleep_time = args.sleep
start_observation_index = args.start_observation_index
n_observations = args.n_observations
n_samples = args.n_samples
burnin_MCMC = args.burnin_MCMC
ABC_algorithm = args.ABC_alg
ABC_steps = args.ABC_steps
results_folder = args.root_folder
observation_folder = args.observation_folder
inference_folder = args.inference_folder
subsample_size = args.subsample_size

# checks
if model not in ("gaussian", "beta", "gamma", "MA2", "AR2") or technique not in ("SM", "SSM", "FP", "True") or \
        inference_technique not in ("exchange", "ABC"):
    raise NotImplementedError

if technique == "FP" and inference_technique == "exchange":
    raise RuntimeError

true_posterior_available = model not in ("Lorenz95", "fullLorenz95", "fullLorenz95smaller")

print("{} model.".format(model))
# set up the default root folder and other values
default_root_folder = {"gaussian": "results/gaussian/",
                       "gamma": "results/gamma/",
                       "beta": "results/beta/",
                       "AR2": "results/AR2/",
                       "MA2": "results/MA2/"}

extract_params_and_weights_fcns = {"gaussian": extract_params_and_weights_from_journal_gaussian,
                                   "gamma": extract_params_and_weights_from_journal_gamma,
                                   "beta": extract_params_and_weights_from_journal_beta,
                                   "AR2": extract_params_and_weights_from_journal_ar2,
                                   "MA2": extract_params_and_weights_from_journal_ma2, }

extract_params_and_weights_from_journal = extract_params_and_weights_fcns[model]

if results_folder is None:
    results_folder = default_root_folder[model]

results_folder = results_folder + '/'
observation_folder = results_folder + observation_folder + '/'
inference_folder = results_folder + inference_folder + '/'

if sleep_time > 0:
    print("Wait for {} minutes...".format(sleep_time))
    sleep(60 * sleep_time)
    print("Done waiting!")

if technique == "True":
    namefile_postfix_no_index = "_true_posterior"
else:
    if inference_technique == "ABC":
        namefile_postfix_no_index = "_{}_{}_steps_{}_n_samples_{}".format(ABC_algorithm, technique, ABC_steps,
                                                                          n_samples)
    elif inference_technique == "exchange":
        namefile_postfix_no_index = "_{}_n_samples_{}".format("exchange", n_samples)

if model == "gaussian":
    mu_bounds = [-10, 10]
    sigma_bounds = [1, 10]
    lower_bounds = np.array([mu_bounds[0], sigma_bounds[0]])
    upper_bounds = np.array([mu_bounds[1], sigma_bounds[1]])
    param_names = [r"$\mu$", r"$\sigma$"]
    xticks = [-10, -5, 0, 5, 10]
    yticks = [1, 4, 7, 10]
elif model == "beta":
    alpha_bounds = [1, 3]
    beta_bounds = [1, 3]
    lower_bounds = np.array([alpha_bounds[0], beta_bounds[0]])
    upper_bounds = np.array([alpha_bounds[1], beta_bounds[1]])
    param_names = [r"$\alpha$", r"$\beta$"]
    xticks = [1, 1.5, 2, 2.5, 3]
    yticks = [1, 1.5, 2, 2.5, 3]
elif model == "gamma":
    k_bounds = [1, 3]
    theta_bounds = [1, 3]
    lower_bounds = np.array([k_bounds[0], theta_bounds[0]])
    upper_bounds = np.array([k_bounds[1], theta_bounds[1]])
    param_names = [r"$k$", r"$\theta$"]
    xticks = [1, 1.5, 2, 2.5, 3]
    yticks = [1, 1.5, 2, 2.5, 3]
elif model == "AR2":
    arma_size = 100
    ar1_bounds = [-1, 1]
    ar2_bounds = [-1, 0]
    lower_bounds = np.array([ar1_bounds[0], ar2_bounds[0]])
    upper_bounds = np.array([ar1_bounds[1], ar2_bounds[1]])
    param_names = [r"$\theta_1$", r"$\theta_2$"]
    xticks = [-1, -0.5, 0, 0.5, 1]
    yticks = [-1, -0.8, -0.6, -0.4, -0.2, 0]
elif model == "MA2":
    arma_size = 100
    ma1_bounds = [-1, 1]
    ma2_bounds = [0, 1]
    lower_bounds = np.array([ma1_bounds[0], ma2_bounds[0]])
    upper_bounds = np.array([ma1_bounds[1], ma2_bounds[1]])
    param_names = [r"$\theta_1$", r"$\theta_2$"]
    xticks = [-1, -0.5, 0, 0.5, 1]
    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]

# sizes for plot
size = 30
linewidth = 4.5
labelsize = 16
for obs_index in range(start_observation_index, n_observations):
    print("Observation ", obs_index + 1)
    namefile_postfix = "_{}".format(obs_index + 1) + namefile_postfix_no_index
    theta_obs = np.load(observation_folder + "theta_obs{}.npy".format(obs_index + 1))

    if technique == "True":
        # load and then subsample the traces otherwise it takes too much
        posterior_samples = np.load(observation_folder + "true_mcmc_trace{}.npy".format(obs_index + 1))
        posterior_samples = subsample_trace(posterior_samples, size=subsample_size)
    else:
        if inference_technique == "exchange":
            posterior_samples = np.load(inference_folder + "exchange_mcmc_trace{}.npy".format(obs_index + 1))
            posterior_samples = subsample_trace(posterior_samples, size=subsample_size)
        elif inference_technique == "ABC":
            jrnl = Journal.fromFile(inference_folder + "jrnl" + namefile_postfix + ".jnl")
            posterior_samples, weights = extract_params_and_weights_from_journal(jrnl)

    fig, ax = plot_bivariate_marginal_with_trace_samples(theta_obs, posterior_samples,
                                                         thetarange=np.array([lower_bounds, upper_bounds]),
                                                         namefile=None, color="#40739E", thresh=0.15,
                                                         figsize_vertical=5, title_size=size, label_size=size,
                                                         param1_name=param_names[0], param2_name=param_names[1],
                                                         space_subplots=0.3, linewidth=linewidth)

    # put titles:
    if technique == "True":
        ax[0].set_title("True posterior", size=size)
    else:
        ax[0].set_title(f"{'ABC' if inference_technique == 'ABC' else 'Exc'}-{technique}", size=size)

    # # if you want to remove ticks and labels:
    # ax[0].tick_params(
    #     axis='both',
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     left=False,
    #     right=False,
    #     labelbottom=False,  # labels along the bottom edge are off
    #     labelleft=False)

    # set size of ticklabels:
    ax[0].tick_params(
        axis='both',
        which='both',  # both major and minor ticks are affected
        labelsize=labelsize, )

    # manually set ticks:
    ax[0].set_xticks(xticks)
    ax[0].set_yticks(yticks)

    # legend size:
    ax[0].legend(fontsize=labelsize)

    if inference_technique == "exchange" and technique != "True":
        namefile = results_folder + f"{technique}_bivariate_marginals" + namefile_postfix + ".pdf"
    else:
        namefile = results_folder + "bivariate_marginals" + namefile_postfix + ".pdf"

    bbox_inches = Bbox(np.array([[-0.48, -0.23], [4.68, 4.8]]))
    plt.savefig(namefile, bbox_inches=bbox_inches)

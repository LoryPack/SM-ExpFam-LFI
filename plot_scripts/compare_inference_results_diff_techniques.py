import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

sys.path.append(os.getcwd())

from abcpy.output import Journal

from src.utils_arma_example import extract_params_and_weights_from_journal_ar2, \
    extract_params_and_weights_from_journal_ma2, extract_posterior_mean_from_journal_ar2, \
    extract_posterior_mean_from_journal_ma2
from src.utils_beta_example import extract_params_and_weights_from_journal_beta, \
    extract_posterior_mean_from_journal_beta
from src.utils_gaussian_example import extract_params_and_weights_from_journal_gaussian, \
    extract_posterior_mean_from_journal_gaussian
from src.utils_gamma_example import extract_params_and_weights_from_journal_gamma, \
    extract_posterior_mean_from_journal_gamma
from src.functions import wass_dist, subsample_trace, \
    plot_confidence_bands_performance_vs_iteration, try_loading_wass_RMSE_post_mean, check_iteration_better_perf

# need to compare the following techniques:
# 1) SL 2) RE 3) ABC FP 4) ABC SM  5) Exchange SM
# comparison will be computing the Wass distance for now

# this can be done for the different models we have

# we need to have a loop over observation that allows to compute the Wass distance for each of them, and then we
# average. Actually, we already computed the Wasserstein distance for the methods and should be stored somewhere.
# However, we may also be interested in the Wass distance obtained after some iterations of ABC or SL/RE?
# Note however that SABC does not always stop at the same iteration, as the time it stops is adaptive I guess.
# note that this routine assumes the Wass distances have been computed correctly for the corresponding files.

# then we need to produce some smart plots for this.

# how much does the estimate for Wass dist depend on the choice of the subsample size?

parser = argparse.ArgumentParser()
parser.add_argument('model',
                    help="The statistical model to consider; can be 'AR2', 'MA2', 'beta', 'gamma', 'gaussian'")
parser.add_argument('plot', help="The plot to produce; can be 'final', 'SL_RE', 'SL', 'RE' or 'ABC'.")
parser.add_argument('--sleep', type=float, default=0, help='Minutes to sleep before starting')
parser.add_argument('--start_observation_index', type=int, default=0, help='Index to start from')
parser.add_argument('--n_observations', type=int, default=10, help='Total number of observations.')
parser.add_argument('--observation_folder', type=str, default="observations")
parser.add_argument('--root_folder', type=str, default=None)
parser.add_argument('--inference_folder_SL', type=str, default="PMC-SL")
parser.add_argument('--inference_folder_RE', type=str, default="PMC-RE")
parser.add_argument('--inference_folder_ABC_FP', type=str, default="ABC-FP")
parser.add_argument('--inference_folder_ABC_SM', type=str, default="ABC-SM")
parser.add_argument('--inference_folder_exchange_SM', type=str, default="Exc-SM")
parser.add_argument('--SL_steps', type=int, default=10)
parser.add_argument('--SL_n_samples', type=int, default=1000)
parser.add_argument('--SL_n_samples_per_param', type=int, default=100)
parser.add_argument('--RE_steps', type=int, default=10)
parser.add_argument('--RE_n_samples', type=int, default=1000)
parser.add_argument('--RE_n_samples_per_param', type=int, default=1000)
parser.add_argument('--ABC_FP_algorithm', type=str, default="SABC")
parser.add_argument('--ABC_FP_n_samples', type=int, default=1000)
parser.add_argument('--ABC_FP_steps', type=int, default=100)
parser.add_argument('--ABC_SM_algorithm', type=str, default="SABC")
parser.add_argument('--ABC_SM_steps', type=int, default=100)
parser.add_argument('--ABC_SM_n_samples', type=int, default=1000)
parser.add_argument('--exchange_SM_n_samples', type=int, default=10000)
parser.add_argument('--load_SL_if_available', action="store_true")
parser.add_argument('--load_RE_if_available', action="store_true")
parser.add_argument('--load_ABC_FP_if_available', action="store_true")
parser.add_argument('--load_ABC_SM_if_available', action="store_true")
parser.add_argument('--load_exchange_SM_if_available', action="store_true")
parser.add_argument('--load_all_if_available', action="store_true")
parser.add_argument('--CI_level', type=float, default=95,
                    help="The size of confidence interval (CI) to produce the plots. It represents the confidence "
                         "intervals in the plots vs iterations, and the position of the whiskers in the boxplots.")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--subsample_size', type=int, default=1000,
                    help='Number of samples used for bivariate plots and Wass distance(if required).')

args = parser.parse_args()

model = args.model
plot = args.plot
sleep_time = args.sleep
start_observation_index = args.start_observation_index
n_observations = args.n_observations
results_folder = args.root_folder
seed = args.seed
subsample_size = args.subsample_size

SL_steps = args.SL_steps
SL_n_samples = args.SL_n_samples
SL_n_samples_per_param = args.SL_n_samples_per_param
RE_steps = args.RE_steps
RE_n_samples = args.RE_n_samples
RE_n_samples_per_param = args.RE_n_samples_per_param
ABC_FP_algorithm = args.ABC_FP_algorithm
ABC_FP_steps = args.ABC_FP_steps
ABC_FP_n_samples = args.ABC_FP_n_samples
ABC_SM_algorithm = args.ABC_SM_algorithm
ABC_SM_steps = args.ABC_SM_steps
ABC_SM_n_samples = args.ABC_SM_n_samples
exchange_SM_n_samples = args.exchange_SM_n_samples
load_SL_if_available = args.load_SL_if_available
load_RE_if_available = args.load_RE_if_available
load_ABC_FP_if_available = args.load_ABC_FP_if_available
load_ABC_SM_if_available = args.load_ABC_SM_if_available
load_exchange_SM_if_available = args.load_exchange_SM_if_available
load_all_if_available = args.load_all_if_available
CI_level = args.CI_level

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
                                   "MA2": extract_params_and_weights_from_journal_ma2, }
extract_posterior_mean_fcns = {"gaussian": extract_posterior_mean_from_journal_gaussian,
                               "gamma": extract_posterior_mean_from_journal_gamma,
                               "beta": extract_posterior_mean_from_journal_beta,
                               "AR2": extract_posterior_mean_from_journal_ar2,
                               "MA2": extract_posterior_mean_from_journal_ma2}
model_text = {"gaussian": "Gaussian", "gamma": "Gamma", "beta": "Beta", "AR2": "AR(2)", "MA2": "MA(2)"}

if results_folder is None:
    results_folder = default_root_folder[model]

inference_folder_SL = results_folder + "/" + args.inference_folder_SL + "/"
inference_folder_RE = results_folder + "/" + args.inference_folder_RE + "/"
inference_folder_ABC_FP = results_folder + "/" + args.inference_folder_ABC_FP + "/"
inference_folder_ABC_SM = results_folder + "/" + args.inference_folder_ABC_SM + "/"
inference_folder_exchange_SM = results_folder + "/" + args.inference_folder_exchange_SM + "/"

observation_folder = results_folder + args.observation_folder + "/"  # the one in the two results folders should be the same.

extract_params_and_weights_from_journal = extract_params_and_weights_fcns[model]
extract_posterior_mean_from_journal = extract_posterior_mean_fcns[model]

if model not in ("gaussian", "beta", "gamma", "MA2", "AR2",) or plot not in ('final', 'SL_RE', 'ABC', 'SL', 'RE'):
    raise NotImplementedError

if load_all_if_available:
    load_SL_if_available = True
    load_RE_if_available = True
    load_ABC_FP_if_available = True
    load_ABC_SM_if_available = True
    load_exchange_SM_if_available = True

true_posterior_available = model in ("gaussian", "beta", "gamma", "MA2", "AR2")

print(f"{model} model.")

SL_namefile_postfix_no_index = f"_steps_{SL_steps}_n_samples_{SL_n_samples}_" \
                               f"n_samples_per_param_{SL_n_samples_per_param}"
RE_namefile_postfix_no_index = f"_steps_{RE_steps}_n_samples_{RE_n_samples}_" \
                               f"n_samples_per_param_{RE_n_samples_per_param}"
ABC_FP_namefile_postfix_no_index = f"_{ABC_FP_algorithm}_FP_steps_{ABC_FP_steps}_n_samples_{ABC_FP_n_samples}"
ABC_SM_namefile_postfix_no_index = f"_{ABC_SM_algorithm}_SM_steps_{ABC_SM_steps}_n_samples_{ABC_SM_n_samples}"

wass_dist_exchange_SM, RMSE_post_mean_exchange_SM, load_successful_exchange_SM = try_loading_wass_RMSE_post_mean(
    inference_folder_exchange_SM, f"_exchange_n_samples_{exchange_SM_n_samples}", load_exchange_SM_if_available)
if not load_successful_exchange_SM:
    wass_dist_exchange_SM = np.zeros(n_observations - start_observation_index)
    RMSE_post_mean_exchange_SM = np.zeros(n_observations - start_observation_index)

if plot == "final":
    # first try loading the Wass distances:
    wass_dist_SL, RMSE_post_mean_SL, load_successful_SL = try_loading_wass_RMSE_post_mean(
        inference_folder_SL, SL_namefile_postfix_no_index, load_SL_if_available)
    wass_dist_RE, RMSE_post_mean_RE, load_successful_RE = try_loading_wass_RMSE_post_mean(
        inference_folder_RE, RE_namefile_postfix_no_index, load_RE_if_available)
    wass_dist_ABC_FP, RMSE_post_mean_ABC_FP, load_successful_ABC_FP = try_loading_wass_RMSE_post_mean(
        inference_folder_ABC_FP, ABC_FP_namefile_postfix_no_index, load_ABC_FP_if_available)
    wass_dist_ABC_SM, RMSE_post_mean_ABC_SM, load_successful_ABC_SM = try_loading_wass_RMSE_post_mean(
        inference_folder_ABC_SM, ABC_SM_namefile_postfix_no_index, load_ABC_SM_if_available)

    # if loading was not successful: need to compute them from scratch
    if not load_successful_SL:
        wass_dist_SL = np.zeros((n_observations - start_observation_index))
        RMSE_post_mean_SL = np.zeros((n_observations - start_observation_index))
    if not load_successful_RE:
        wass_dist_RE = np.zeros((n_observations - start_observation_index))
        RMSE_post_mean_RE = np.zeros((n_observations - start_observation_index))
    if not load_successful_ABC_FP:
        wass_dist_ABC_FP = np.zeros((n_observations - start_observation_index))
        RMSE_post_mean_ABC_FP = np.zeros((n_observations - start_observation_index))
    if not load_successful_ABC_SM:
        wass_dist_ABC_SM = np.zeros((n_observations - start_observation_index))
        RMSE_post_mean_ABC_SM = np.zeros((n_observations - start_observation_index))

    for obs_index in range(start_observation_index, n_observations):
        print("Observation ", obs_index + 1)
        x_obs = np.load(observation_folder + f"x_obs{obs_index + 1}.npy")
        theta_obs = np.load(observation_folder + f"theta_obs{obs_index + 1}.npy")
        if true_posterior_available:
            trace_true = np.load(observation_folder + f"true_mcmc_trace{obs_index + 1}.npy")
            trace_true_subsample = subsample_trace(trace_true, size=subsample_size)
            # crashes if subsample_size>n_samples.
            true_post_means = np.mean(trace_true, axis=0)

        if true_posterior_available and not load_successful_exchange_SM:
            # load the exchange MCMC trace and compute Wass dist:
            trace_exchange = np.load(inference_folder_exchange_SM + f"exchange_mcmc_trace{obs_index + 1}.npy")
            means_exchange = np.mean(trace_exchange, axis=0)
            trace_exchange_subsample = subsample_trace(trace_exchange,
                                                       size=subsample_size)  # used to compute wass dist
            # compute wass distance and RMSE
            wass_dist_exchange_SM[obs_index] = wass_dist(trace_exchange_subsample, trace_true_subsample,
                                                         numItermax=10 ** 6)
            RMSE_post_mean_exchange_SM[obs_index] = np.linalg.norm(means_exchange - true_post_means)

            np.save(inference_folder_exchange_SM + "wass_dist", wass_dist_exchange_SM)
            np.save(inference_folder_exchange_SM + "RMSE_post_mean", RMSE_post_mean_exchange_SM)

        if true_posterior_available:
            # wass dist and RMSE:
            if not load_successful_SL:
                SL_namefile_postfix = f"_{obs_index + 1}" + SL_namefile_postfix_no_index
                jrnl_SL = Journal.fromFile(inference_folder_SL + "jrnl" + SL_namefile_postfix + ".jnl")
                params_SL, weights_SL = extract_params_and_weights_from_journal(jrnl_SL, )
                means_SL = extract_posterior_mean_from_journal(jrnl_SL, )
                wass_dist_SL[obs_index] = wass_dist(params_SL, trace_true_subsample,
                                                    weights_post_1=weights_SL, numItermax=10 ** 6)
                RMSE_post_mean_SL[obs_index] = np.linalg.norm(means_SL - true_post_means)
                np.save(inference_folder_SL + "wass_dist" + SL_namefile_postfix_no_index, wass_dist_SL)
                np.save(inference_folder_SL + "RMSE_post_mean" + SL_namefile_postfix_no_index,
                        RMSE_post_mean_SL)
            if not load_successful_RE:
                RE_namefile_postfix = f"_{obs_index + 1}" + RE_namefile_postfix_no_index
                jrnl_RE = Journal.fromFile(inference_folder_RE + "jrnl" + RE_namefile_postfix + ".jnl")
                params_RE, weights_RE = extract_params_and_weights_from_journal(jrnl_RE, )
                means_RE = extract_posterior_mean_from_journal(jrnl_RE, )
                wass_dist_RE[obs_index] = wass_dist(params_RE, trace_true_subsample,
                                                    weights_post_1=weights_RE, numItermax=10 ** 6)
                RMSE_post_mean_RE[obs_index] = np.linalg.norm(means_RE - true_post_means)
                np.save(inference_folder_RE + "wass_dist" + RE_namefile_postfix_no_index, wass_dist_RE)
                np.save(inference_folder_RE + "RMSE_post_mean" + RE_namefile_postfix_no_index,
                        RMSE_post_mean_RE)
            if not load_successful_ABC_FP:
                ABC_FP_namefile_postfix = f"_{obs_index + 1}" + ABC_FP_namefile_postfix_no_index
                jrnl_ABC_FP = Journal.fromFile(inference_folder_ABC_FP + "jrnl" + ABC_FP_namefile_postfix + ".jnl")
                params_ABC_FP, weights_ABC_FP = extract_params_and_weights_from_journal(jrnl_ABC_FP, )
                means_ABC_FP = extract_posterior_mean_from_journal(jrnl_ABC_FP, )

                wass_dist_ABC_FP[obs_index] = wass_dist(params_ABC_FP, trace_true_subsample,
                                                        weights_post_1=weights_ABC_FP, numItermax=10 ** 6)
                RMSE_post_mean_ABC_FP[obs_index] = np.linalg.norm(means_ABC_FP - true_post_means)
                np.save(inference_folder_ABC_FP + "wass_dist" + ABC_FP_namefile_postfix_no_index,
                        wass_dist_ABC_FP)
                np.save(inference_folder_ABC_FP + "RMSE_post_mean" + ABC_FP_namefile_postfix_no_index,
                        RMSE_post_mean_ABC_FP)
            if not load_successful_ABC_SM:
                # now need to load the journal files and then compute the Wass dist for each iteration:
                ABC_SM_namefile_postfix = f"_{obs_index + 1}" + ABC_SM_namefile_postfix_no_index
                jrnl_ABC_SM = Journal.fromFile(inference_folder_ABC_SM + "jrnl" + ABC_SM_namefile_postfix + ".jnl")
                params_ABC_SM, weights_ABC_SM = extract_params_and_weights_from_journal(jrnl_ABC_SM, )
                means_ABC_SM = extract_posterior_mean_from_journal(jrnl_ABC_SM, )

                wass_dist_ABC_SM[obs_index] = wass_dist(params_ABC_SM, trace_true_subsample,
                                                        weights_post_1=weights_ABC_SM, numItermax=10 ** 6)
                RMSE_post_mean_ABC_SM[obs_index] = np.linalg.norm(means_ABC_SM - true_post_means)
                np.save(inference_folder_ABC_SM + "wass_dist" + ABC_SM_namefile_postfix_no_index,
                        wass_dist_ABC_SM)
                np.save(inference_folder_ABC_SM + "RMSE_post_mean" + ABC_SM_namefile_postfix_no_index,
                        RMSE_post_mean_ABC_SM)

    list_names = ["PMC-SL", "PMC-RE", "ABC-FP", "ABC-SM", "Exchange SM"]
    list_names_short = ["PMC-SL", "PMC-RE", "ABC-FP", "ABC-SM", "Exc-SM"]
    list_indeces = [1, 2, 3, 4, 5]
    list_wass_dist = [wass_dist_SL, wass_dist_RE, wass_dist_ABC_FP, wass_dist_ABC_SM, wass_dist_exchange_SM]
    list_RMSE_post_mean = [RMSE_post_mean_SL, RMSE_post_mean_RE, RMSE_post_mean_ABC_FP, RMSE_post_mean_ABC_SM,
                           RMSE_post_mean_exchange_SM]

    fig = plt.figure(figsize=(2 * len(list_names), 6))
    # boxplots: the box covers range from 1st to 3rd quartile. We set the whiskers to the same band size in the
    # confidence intervals plots
    plt.boxplot(list_wass_dist, whis=(50 - CI_level / 2, 50 + CI_level / 2))
    plt.xticks(list_indeces, list_names)
    plt.ylabel("Wasserstein distance")
    # plt.show()
    plt.savefig(results_folder + "wass_dist.pdf")
    plt.close()

    fig = plt.figure(figsize=(2 * len(list_names), 6))
    plt.boxplot(list_RMSE_post_mean, whis=(50 - CI_level / 2, 50 + CI_level / 2))
    plt.xticks(list_indeces, list_names)
    plt.ylabel("RMSE posterior mean")
    # plt.show()
    plt.savefig(results_folder + "RMSE_post_mean.pdf")
    plt.close()

    # joint plot:
    fig, ax1 = plt.subplots(1, figsize=(0.8 * len(list_names) + 1, 6))
    ax2 = ax1.twinx()
    ax1.set_title(f"{model_text[model]}")
    c = "blue"
    ax1.boxplot(list_wass_dist, whis=(50 - CI_level / 2, 50 + CI_level / 2), patch_artist=True, notch=True,
                boxprops=dict(facecolor=c, color=c, alpha=0.5),
                capprops=dict(color=c),
                whiskerprops=dict(color=c),
                flierprops=dict(color=c, markeredgecolor=c),
                medianprops=dict(color=c), widths=0.3, positions=np.array(list_indeces) - 0.15)
    ax1.set_xticks(list_indeces)
    ax1.tick_params(axis='y', labelcolor=c)
    ax1.set_xticklabels(list_names_short, rotation=45)
    ax1.set_ylabel("Wasserstein distance", color=c)
    ax1.set_ylim(ymin=0)
    c = "red"
    ax2.boxplot(list_RMSE_post_mean, whis=(50 - CI_level / 2, 50 + CI_level / 2), patch_artist=True, notch=True,
                boxprops=dict(facecolor=c, color=c, alpha=0.5),
                capprops=dict(color=c),
                whiskerprops=dict(color=c),
                flierprops=dict(color=c, markeredgecolor=c),
                medianprops=dict(color=c), widths=0.3, positions=np.array(list_indeces) + 0.15)
    ax2.set_ylabel("RMSE posterior mean", color=c)
    ax2.tick_params(axis='y', labelcolor=c)
    ax2.set_xticks(list_indeces)
    ax2.set_xticklabels(list_names_short, rotation=45)
    ax2.set_ylim(ymin=0)
    bbox_inches = Bbox(np.array([[-0.1, -0.15], [0.8 * len(list_names) + 1.2, 5.6]]))
    plt.savefig(results_folder + "joint_boxplot.pdf", bbox_inches=bbox_inches)
    plt.close()
else:
    bbox_inches = Bbox(np.array([[-0.1, -0.15], [5.5, 3.8]]))
    if plot == "SL_RE":

        # first try loading the Wass distances:
        wass_dist_SL, RMSE_post_mean_SL, load_successful_SL = try_loading_wass_RMSE_post_mean(
            inference_folder_SL, "_iterations" + SL_namefile_postfix_no_index, load_SL_if_available)
        wass_dist_RE, RMSE_post_mean_RE, load_successful_RE = try_loading_wass_RMSE_post_mean(
            inference_folder_RE, "_iterations" + RE_namefile_postfix_no_index, load_RE_if_available)

        # if loading was not successful: need to compute them from scratch
        if not load_successful_SL:
            wass_dist_SL = np.zeros((SL_steps, n_observations - start_observation_index))
            RMSE_post_mean_SL = np.zeros((SL_steps, n_observations - start_observation_index))
        if not load_successful_RE:
            wass_dist_RE = np.zeros((RE_steps, n_observations - start_observation_index))
            RMSE_post_mean_RE = np.zeros((RE_steps, n_observations - start_observation_index))

        for obs_index in range(start_observation_index, n_observations):
            print("Observation ", obs_index + 1)
            x_obs = np.load(observation_folder + f"x_obs{obs_index + 1}.npy")
            theta_obs = np.load(observation_folder + f"theta_obs{obs_index + 1}.npy")
            if true_posterior_available:
                trace_true = np.load(observation_folder + f"true_mcmc_trace{obs_index + 1}.npy")
                trace_true_subsample = subsample_trace(trace_true, size=subsample_size)
                # crashes if subsample_size>n_samples.
                true_post_means = np.mean(trace_true, axis=0)

            if true_posterior_available and not load_successful_exchange_SM:
                # load the exchange MCMC trace and compute Wass dist:
                trace_exchange = np.load(inference_folder_exchange_SM + f"exchange_mcmc_trace{obs_index + 1}.npy")
                means_exchange = np.mean(trace_exchange, axis=0)
                trace_exchange_subsample = subsample_trace(trace_exchange,
                                                           size=subsample_size)  # used to compute wass dist
                # compute wass distance and RMSE
                wass_dist_exchange_SM[obs_index] = wass_dist(trace_exchange_subsample, trace_true_subsample,
                                                             numItermax=10 ** 6)
                RMSE_post_mean_exchange_SM[obs_index] = np.linalg.norm(means_exchange - true_post_means)

                np.save(inference_folder_exchange_SM + "wass_dist", wass_dist_exchange_SM)
                np.save(inference_folder_exchange_SM + "RMSE_post_mean", RMSE_post_mean_exchange_SM)

            # now need to load the journal files and then compute the Wass dist for each iteration:
            if not load_successful_SL:
                SL_namefile_postfix = f"_{obs_index + 1}" + SL_namefile_postfix_no_index
                jrnl_SL = Journal.fromFile(inference_folder_SL + "jrnl" + SL_namefile_postfix + ".jnl")
            if not load_successful_RE:
                RE_namefile_postfix = f"_{obs_index + 1}" + RE_namefile_postfix_no_index
                jrnl_RE = Journal.fromFile(inference_folder_RE + "jrnl" + RE_namefile_postfix + ".jnl")

            for iteration in range(SL_steps):  # assumes SL_steps == RE_steps
                if true_posterior_available:
                    # wass dist and RMSE:
                    if not load_successful_SL:
                        params_SL, weights_SL = extract_params_and_weights_from_journal(jrnl_SL, step=iteration)
                        means_SL = extract_posterior_mean_from_journal(jrnl_SL, step=iteration)
                        wass_dist_SL[iteration, obs_index] = wass_dist(params_SL, trace_true_subsample,
                                                                       weights_post_1=weights_SL, numItermax=10 ** 6)
                        RMSE_post_mean_SL[iteration, obs_index] = np.linalg.norm(means_SL - true_post_means)
                        np.save(inference_folder_SL + "wass_dist_iterations" + SL_namefile_postfix_no_index,
                                wass_dist_SL)
                        np.save(inference_folder_SL + "RMSE_post_mean_iterations" + SL_namefile_postfix_no_index,
                                RMSE_post_mean_SL)
                    if not load_successful_RE:
                        params_RE, weights_RE = extract_params_and_weights_from_journal(jrnl_RE, step=iteration)
                        means_RE = extract_posterior_mean_from_journal(jrnl_RE, step=iteration)
                        wass_dist_RE[iteration, obs_index] = wass_dist(params_RE, trace_true_subsample,
                                                                       weights_post_1=weights_RE, numItermax=10 ** 6)
                        RMSE_post_mean_RE[iteration, obs_index] = np.linalg.norm(means_RE - true_post_means)
                        np.save(inference_folder_RE + "wass_dist_iterations" + RE_namefile_postfix_no_index,
                                wass_dist_RE)
                        np.save(inference_folder_RE + "RMSE_post_mean_iterations" + RE_namefile_postfix_no_index,
                                RMSE_post_mean_RE)
        # now create the plot!
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
        plot_confidence_bands_performance_vs_iteration(wass_dist_SL, fig=fig, ax=ax, label="PMC-SL", color_band_1="C1",
                                                       color_line="C1", band_1=CI_level, alpha_1=0.3, band_2=0,
                                                       band_3=0,
                                                       hatch='/')
        plot_confidence_bands_performance_vs_iteration(wass_dist_RE, fig=fig, ax=ax, label="PMC-RE", color_band_1="C2",
                                                       color_line="C2", band_1=CI_level, alpha_1=0.3, band_2=0,
                                                       band_3=0,
                                                       hatch='\\')
        # stack the following in order to have an horizontal plot:
        plot_confidence_bands_performance_vs_iteration(np.stack([wass_dist_exchange_SM] * max(SL_steps, RE_steps)),
                                                       fig=fig,
                                                       ax=ax, label="Exc-SM", color_band_1="blue", color_line="blue",
                                                       band_1=CI_level, alpha_1=0.3, band_2=0, band_3=0,
                                                       hatch='.')
        ax.set_title(f"{model_text[model]}")
        ax.set_ylabel("Wasserstein distance")
        ax.set_xlabel("Iteration")
        # ax.set_xticks([1,3, 5, 7, 9])
        # ax.set_xticks([0, 2, 4, 6, 8], minor=True)
        # ax.set_xticklabels([200, 4, 6, 8, 10])
        ax.legend()
        plt.savefig(results_folder + "wass_dist_SL_RE_iterations.pdf", bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
        plot_confidence_bands_performance_vs_iteration(RMSE_post_mean_SL, fig=fig, ax=ax, label="PMC-SL",
                                                       color_band_1="C1",
                                                       color_line="C1", band_1=CI_level, alpha_1=0.3, band_2=0,
                                                       band_3=0,
                                                       hatch='/')
        plot_confidence_bands_performance_vs_iteration(RMSE_post_mean_RE, fig=fig, ax=ax, label="PMC-RE",
                                                       color_band_1="C2",
                                                       color_line="C2", band_1=CI_level, alpha_1=0.3, band_2=0,
                                                       band_3=0,
                                                       hatch='\\')
        # stack the following in order to have an horizontal plot:
        plot_confidence_bands_performance_vs_iteration(np.stack([RMSE_post_mean_exchange_SM] * max(SL_steps, RE_steps)),
                                                       fig=fig, ax=ax, label="Exc-SM", color_band_1="blue",
                                                       color_line="blue", band_1=CI_level, alpha_1=0.3,
                                                       band_2=0,
                                                       band_3=0, hatch='.')
        ax.set_title("RMSE posterior mean")
        ax.set_xlabel("Iteration")
        ax.legend()
        plt.savefig(results_folder + "RMSE_post_mean_SL_RE_iterations.pdf", bbox_inches="tight")
        plt.close()

    elif plot == "SL":

        # first try loading the Wass distances:
        wass_dist_SL, RMSE_post_mean_SL, load_successful_SL = try_loading_wass_RMSE_post_mean(
            inference_folder_SL, "_iterations" + SL_namefile_postfix_no_index, load_SL_if_available)

        # if loading was not successful: need to compute them from scratch
        if not load_successful_SL:
            wass_dist_SL = np.zeros((SL_steps, n_observations - start_observation_index))
            RMSE_post_mean_SL = np.zeros((SL_steps, n_observations - start_observation_index))

        for obs_index in range(start_observation_index, n_observations):
            print("Observation ", obs_index + 1)
            x_obs = np.load(observation_folder + f"x_obs{obs_index + 1}.npy")
            theta_obs = np.load(observation_folder + f"theta_obs{obs_index + 1}.npy")
            if true_posterior_available:
                trace_true = np.load(observation_folder + f"true_mcmc_trace{obs_index + 1}.npy")
                trace_true_subsample = subsample_trace(trace_true, size=subsample_size)
                # crashes if subsample_size>n_samples.
                true_post_means = np.mean(trace_true, axis=0)

            if true_posterior_available and not load_successful_exchange_SM:
                # load the exchange MCMC trace and compute Wass dist:
                trace_exchange = np.load(inference_folder_exchange_SM + f"exchange_mcmc_trace{obs_index + 1}.npy")
                means_exchange = np.mean(trace_exchange, axis=0)
                trace_exchange_subsample = subsample_trace(trace_exchange,
                                                           size=subsample_size)  # used to compute wass dist
                # compute wass distance and RMSE
                wass_dist_exchange_SM[obs_index] = wass_dist(trace_exchange_subsample, trace_true_subsample,
                                                             numItermax=10 ** 6)
                RMSE_post_mean_exchange_SM[obs_index] = np.linalg.norm(means_exchange - true_post_means)

                np.save(inference_folder_exchange_SM + "wass_dist", wass_dist_exchange_SM)
                np.save(inference_folder_exchange_SM + "RMSE_post_mean", RMSE_post_mean_exchange_SM)

            # now need to load the journal files and then compute the Wass dist for each iteration:
            if not load_successful_SL:
                SL_namefile_postfix = f"_{obs_index + 1}" + SL_namefile_postfix_no_index
                jrnl_SL = Journal.fromFile(inference_folder_SL + "jrnl" + SL_namefile_postfix + ".jnl")

            for iteration in range(SL_steps):  # assumes SL_steps == RE_steps
                if true_posterior_available:
                    # wass dist and RMSE:
                    if not load_successful_SL:
                        params_SL, weights_SL = extract_params_and_weights_from_journal(jrnl_SL, step=iteration)
                        means_SL = extract_posterior_mean_from_journal(jrnl_SL, step=iteration)
                        wass_dist_SL[iteration, obs_index] = wass_dist(params_SL, trace_true_subsample,
                                                                       weights_post_1=weights_SL, numItermax=10 ** 6)
                        RMSE_post_mean_SL[iteration, obs_index] = np.linalg.norm(means_SL - true_post_means)
                        np.save(inference_folder_SL + "wass_dist_iterations" + SL_namefile_postfix_no_index,
                                wass_dist_SL)
                        np.save(inference_folder_SL + "RMSE_post_mean_iterations" + SL_namefile_postfix_no_index,
                                RMSE_post_mean_SL)
        # now create the plot!
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
        plot_confidence_bands_performance_vs_iteration(wass_dist_SL, fig=fig, ax=ax, label="PMC-SL", color_band_1="C1",
                                                       color_line="C1", band_1=CI_level, alpha_1=0.3, alpha_2=0,
                                                       alpha_3=0, fill_between=True)
        # stack the following in order to have an horizontal plot:
        plot_confidence_bands_performance_vs_iteration(np.stack([wass_dist_exchange_SM] * SL_steps), fig=fig,
                                                       ax=ax, label="Exc-SM", color_band_1="blue", color_line="blue",
                                                       band_1=CI_level, alpha_1=1, alpha_2=0, alpha_3=0,
                                                       fill_between=False, ls="--", ls_band_1=":")
        ax.set_title(f"{model_text[model]}")
        ax.set_ylabel("Wasserstein distance")
        # ax.set_xlabel("Iteration")
        ax.set_xlabel(r"Number of simulations ($\times 1000$)")
        ax.set_xticks([1, 3, 5, 7, 9])
        ax.set_xticks([0, 2, 4, 6, 8], minor=True)
        # ax.set_xticklabels([200000, 400000, 600000, 800000, 1000000])
        ax.set_xticklabels([200, 400, 600, 800, 1000])
        ax.set_ylim(ymin=0)
        ax.legend()
        plt.savefig(results_folder + "wass_dist_SL_iterations.pdf", bbox_inches=bbox_inches)
        plt.close()

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
        plot_confidence_bands_performance_vs_iteration(RMSE_post_mean_SL, fig=fig, ax=ax, label="PMC-SL",
                                                       color_band_1="C1",
                                                       color_line="C1", band_1=CI_level, alpha_1=0.3, alpha_2=0,
                                                       alpha_3=0, fill_between=True)
        # stack the following in order to have an horizontal plot:
        plot_confidence_bands_performance_vs_iteration(np.stack([RMSE_post_mean_exchange_SM] * SL_steps),
                                                       fig=fig, ax=ax, label="Exc-SM", color_band_1="blue",
                                                       color_line="blue", band_1=CI_level, alpha_1=1,
                                                       alpha_2=0, alpha_3=0, fill_between=False, ls="--", ls_band_1=":")
        ax.set_title(f"{model_text[model]}")
        ax.set_ylabel(r"RMSE posterior mean")
        # ax.set_xlabel("Iteration")
        ax.set_xlabel(r"Number of simulations ($\times 1000$)")
        ax.set_xticks([1, 3, 5, 7, 9])
        ax.set_xticks([0, 2, 4, 6, 8], minor=True)
        ax.set_xticklabels([200, 400, 600, 800, 1000])
        ax.set_ylim(ymin=0)
        ax.legend()
        plt.savefig(results_folder + "RMSE_post_mean_SL_iterations.pdf", bbox_inches=bbox_inches)
        plt.close()

        # check now at which iteration wass_dist_SL improves upon my method:
        iteration = check_iteration_better_perf(wass_dist_SL, wass_dist_exchange_SM)
        print("SL performs better than my method at iteration: ", None if iteration is None else iteration + 1)

    elif plot == "RE":

        # first try loading the Wass distances:
        wass_dist_RE, RMSE_post_mean_RE, load_successful_RE = try_loading_wass_RMSE_post_mean(
            inference_folder_RE, "_iterations" + RE_namefile_postfix_no_index, load_RE_if_available)

        # if loading was not successful: need to compute them from scratch
        if not load_successful_RE:
            wass_dist_RE = np.zeros((RE_steps, n_observations - start_observation_index))
            RMSE_post_mean_RE = np.zeros((RE_steps, n_observations - start_observation_index))

        for obs_index in range(start_observation_index, n_observations):
            print("Observation ", obs_index + 1)
            x_obs = np.load(observation_folder + f"x_obs{obs_index + 1}.npy")
            theta_obs = np.load(observation_folder + f"theta_obs{obs_index + 1}.npy")
            if true_posterior_available:
                trace_true = np.load(observation_folder + f"true_mcmc_trace{obs_index + 1}.npy")
                trace_true_subsample = subsample_trace(trace_true, size=subsample_size)
                # crashes if subsample_size>n_samples.
                true_post_means = np.mean(trace_true, axis=0)

            if true_posterior_available and not load_successful_exchange_SM:
                # load the exchange MCMC trace and compute Wass dist:
                trace_exchange = np.load(inference_folder_exchange_SM + f"exchange_mcmc_trace{obs_index + 1}.npy")
                means_exchange = np.mean(trace_exchange, axis=0)
                trace_exchange_subsample = subsample_trace(trace_exchange,
                                                           size=subsample_size)  # used to compute wass dist
                # compute wass distance and RMSE
                wass_dist_exchange_SM[obs_index] = wass_dist(trace_exchange_subsample, trace_true_subsample,
                                                             numItermax=10 ** 6)
                RMSE_post_mean_exchange_SM[obs_index] = np.linalg.norm(means_exchange - true_post_means)

                np.save(inference_folder_exchange_SM + "wass_dist", wass_dist_exchange_SM)
                np.save(inference_folder_exchange_SM + "RMSE_post_mean", RMSE_post_mean_exchange_SM)

            # now need to load the journal files and then compute the Wass dist for each iteration:
            if not load_successful_RE:
                RE_namefile_postfix = f"_{obs_index + 1}" + RE_namefile_postfix_no_index
                jrnl_RE = Journal.fromFile(inference_folder_RE + "jrnl" + RE_namefile_postfix + ".jnl")

            for iteration in range(RE_steps):  # assumes SL_steps == RE_steps
                if true_posterior_available:
                    # wass dist and RMSE:
                    if not load_successful_RE:
                        params_RE, weights_RE = extract_params_and_weights_from_journal(jrnl_RE, step=iteration)
                        means_RE = extract_posterior_mean_from_journal(jrnl_RE, step=iteration)
                        wass_dist_RE[iteration, obs_index] = wass_dist(params_RE, trace_true_subsample,
                                                                       weights_post_1=weights_RE, numItermax=10 ** 6)
                        RMSE_post_mean_RE[iteration, obs_index] = np.linalg.norm(means_RE - true_post_means)
                        np.save(inference_folder_RE + "wass_dist_iterations" + RE_namefile_postfix_no_index,
                                wass_dist_RE)
                        np.save(inference_folder_RE + "RMSE_post_mean_iterations" + RE_namefile_postfix_no_index,
                                RMSE_post_mean_RE)
        # now create the plot!
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
        plot_confidence_bands_performance_vs_iteration(wass_dist_RE, fig=fig, ax=ax, label="PMC-RE", color_band_1="C2",
                                                       color_line="C2", band_1=CI_level, alpha_1=0.3, alpha_2=0,
                                                       alpha_3=0, fill_between=True)
        # stack the following in order to have an horizontal plot:
        plot_confidence_bands_performance_vs_iteration(np.stack([wass_dist_exchange_SM] * SL_steps), fig=fig,
                                                       ax=ax, label="Exc-SM", color_band_1="blue", color_line="blue",
                                                       band_1=CI_level, alpha_1=1, alpha_2=0, alpha_3=0,
                                                       fill_between=False, ls="--", ls_band_1=":")
        ax.set_title(f"{model_text[model]}")
        ax.set_ylabel("Wasserstein distance")
        # ax.set_xlabel("Iteration")
        ax.set_xlabel(r"Number of simulations ($\times 1000$)")
        ax.set_xticks([1, 3, 5, 7, 9])
        ax.set_xticks([0, 2, 4, 6, 8], minor=True)
        ax.set_xticklabels([2000, 4000, 6000, 8000, 10000])
        ax.set_ylim(ymin=0)
        ax.legend()
        plt.savefig(results_folder + "wass_dist_RE_iterations.pdf", bbox_inches=bbox_inches)
        plt.close()

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
        plot_confidence_bands_performance_vs_iteration(RMSE_post_mean_RE, fig=fig, ax=ax, label="PMC-RE",
                                                       color_band_1="C2",
                                                       color_line="C2", band_1=CI_level, alpha_1=0.3, alpha_2=0,
                                                       alpha_3=0, fill_between=True)
        # stack the following in order to have an horizontal plot:
        plot_confidence_bands_performance_vs_iteration(np.stack([RMSE_post_mean_exchange_SM] * RE_steps),
                                                       fig=fig, ax=ax, label="Exc-SM", color_band_1="blue",
                                                       color_line="blue", band_1=CI_level, alpha_1=1,
                                                       alpha_2=0, alpha_3=0, fill_between=False, ls="--", ls_band_1=":")
        ax.set_title(f"{model_text[model]}")
        ax.set_ylabel("RMSE posterior mean")
        # ax.set_xlabel("Iteration")
        ax.set_xlabel(r"Number of simulations ($\times 1000$)")
        ax.set_xticks([1, 3, 5, 7, 9])
        ax.set_xticks([0, 2, 4, 6, 8], minor=True)
        ax.set_xticklabels([2000, 4000, 6000, 8000, 10000])
        ax.set_ylim(ymin=0)
        ax.legend()
        plt.savefig(results_folder + "RMSE_post_mean_RE_iterations.pdf", bbox_inches=bbox_inches)
        plt.close()

        iteration = check_iteration_better_perf(wass_dist_RE, wass_dist_exchange_SM)
        print("RE performs better than my method at iteration: ", None if iteration is None else iteration + 1)

    elif plot in ("ABC",):
        # first try loading the Wass distances:
        wass_dist_ABC_FP, RMSE_post_mean_ABC_FP, load_successful_ABC_FP = try_loading_wass_RMSE_post_mean(
            inference_folder_ABC_FP, "_iterations" + ABC_FP_namefile_postfix_no_index, load_ABC_FP_if_available)
        wass_dist_ABC_SM, RMSE_post_mean_ABC_SM, load_successful_ABC_SM = try_loading_wass_RMSE_post_mean(
            inference_folder_ABC_SM, "_iterations" + ABC_SM_namefile_postfix_no_index, load_ABC_SM_if_available)

        # if loading was not successful: need to compute them from scratch
        if not load_successful_ABC_FP:
            wass_dist_ABC_FP = np.zeros((ABC_FP_steps, n_observations - start_observation_index))
            RMSE_post_mean_ABC_FP = np.zeros((ABC_FP_steps, n_observations - start_observation_index))
        if not load_successful_ABC_SM:
            wass_dist_ABC_SM = np.zeros((ABC_SM_steps, n_observations - start_observation_index))
            RMSE_post_mean_ABC_SM = np.zeros((ABC_SM_steps, n_observations - start_observation_index))

        for obs_index in range(start_observation_index, n_observations):
            print("Observation ", obs_index + 1)
            x_obs = np.load(observation_folder + f"x_obs{obs_index + 1}.npy")
            theta_obs = np.load(observation_folder + f"theta_obs{obs_index + 1}.npy")
            if true_posterior_available:
                trace_true = np.load(observation_folder + f"true_mcmc_trace{obs_index + 1}.npy")
                trace_true_subsample = subsample_trace(trace_true, size=subsample_size)
                # crashes if subsample_size>n_samples.
                true_post_means = np.mean(trace_true, axis=0)

            if true_posterior_available and not load_successful_exchange_SM:
                # load the exchange MCMC trace and compute Wass dist:
                trace_exchange = np.load(inference_folder_exchange_SM + f"exchange_mcmc_trace{obs_index + 1}.npy")
                means_exchange = np.mean(trace_exchange, axis=0)
                trace_exchange_subsample = subsample_trace(trace_exchange,
                                                           size=subsample_size)  # used to compute wass dist
                # compute wass distance and RMSE
                wass_dist_exchange_SM[obs_index] = wass_dist(trace_exchange_subsample, trace_true_subsample,
                                                             numItermax=10 ** 6)
                RMSE_post_mean_exchange_SM[obs_index] = np.linalg.norm(means_exchange - true_post_means)

                np.save(inference_folder_exchange_SM + "wass_dist", wass_dist_exchange_SM)
                np.save(inference_folder_exchange_SM + "RMSE_post_mean", RMSE_post_mean_exchange_SM)

            # now need to load the journal files and then compute the Wass dist for each iteration:
            if not load_successful_ABC_FP:
                ABC_FP_namefile_postfix = f"_{obs_index + 1}" + ABC_FP_namefile_postfix_no_index
                jrnl_ABC_FP = Journal.fromFile(inference_folder_ABC_FP + "jrnl" + ABC_FP_namefile_postfix + ".jnl")
            if not load_successful_ABC_SM:
                ABC_SM_namefile_postfix = f"_{obs_index + 1}" + ABC_SM_namefile_postfix_no_index
                jrnl_ABC_SM = Journal.fromFile(inference_folder_ABC_SM + "jrnl" + ABC_SM_namefile_postfix + ".jnl")

            for iteration in range(ABC_FP_steps):  # assume the iterations for ABC_FP and ABC_SM are the same
                if true_posterior_available:
                    # wass dist and RMSE:
                    if not load_successful_ABC_FP:
                        params_ABC_FP, weights_ABC_FP = extract_params_and_weights_from_journal(jrnl_ABC_FP,
                                                                                                step=iteration)
                        means_ABC_FP = extract_posterior_mean_from_journal(jrnl_ABC_FP, step=iteration)

                        wass_dist_ABC_FP[iteration, obs_index] = wass_dist(params_ABC_FP,
                                                                           trace_true_subsample,
                                                                           weights_post_1=weights_ABC_FP,
                                                                           numItermax=10 ** 6)
                        RMSE_post_mean_ABC_FP[iteration, obs_index] = np.linalg.norm(means_ABC_FP - true_post_means)
                        np.save(inference_folder_ABC_FP + "wass_dist_iterations" + ABC_FP_namefile_postfix_no_index,
                                wass_dist_ABC_FP)
                        np.save(
                            inference_folder_ABC_FP + "RMSE_post_mean_iterations" + ABC_FP_namefile_postfix_no_index,
                            RMSE_post_mean_ABC_FP)
                    if not load_successful_ABC_SM:
                        params_ABC_SM, weights_ABC_SM = extract_params_and_weights_from_journal(jrnl_ABC_SM,
                                                                                                step=iteration)
                        means_ABC_SM = extract_posterior_mean_from_journal(jrnl_ABC_SM, step=iteration)
                        wass_dist_ABC_SM[iteration, obs_index] = wass_dist(params_ABC_SM,
                                                                           trace_true_subsample, numItermax=10 ** 6,
                                                                           weights_post_1=weights_ABC_SM)
                        RMSE_post_mean_ABC_SM[iteration, obs_index] = np.linalg.norm(means_ABC_SM - true_post_means)
                        np.save(inference_folder_ABC_SM + "wass_dist_iterations" + ABC_SM_namefile_postfix_no_index,
                                wass_dist_ABC_SM)
                        np.save(
                            inference_folder_ABC_SM + "RMSE_post_mean_iterations" + ABC_SM_namefile_postfix_no_index,
                            RMSE_post_mean_ABC_SM)

        # now create the plot!
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
        plot_confidence_bands_performance_vs_iteration(wass_dist_ABC_FP, fig=fig, ax=ax, label="ABC-FP",
                                                       color_band_1="C1",
                                                       color_line="C1", band_1=CI_level, alpha_1=0.3, alpha_2=0,
                                                       alpha_3=0, fill_between=True)
        plot_confidence_bands_performance_vs_iteration(wass_dist_ABC_SM, fig=fig, ax=ax, label="ABC-SM",
                                                       color_band_1="C2",
                                                       color_line="C2", band_1=CI_level, alpha_1=0.3, alpha_2=0,
                                                       alpha_3=0, fill_between=True)
        plot_confidence_bands_performance_vs_iteration(
            np.stack([wass_dist_exchange_SM] * max(ABC_FP_steps, ABC_SM_steps)),
            fig=fig, ax=ax,
            label="Exc-SM", color_band_1="blue", color_line="blue",
            band_1=CI_level, alpha_1=1, alpha_2=0, alpha_3=0, hatch='.',
            fill_between=False, ls="--", ls_band_1=":")
        # ax.set_title("Wasserstein distance")
        ax.set_title(f"{model_text[model]}")
        ax.set_ylabel("Wasserstein distance")
        ax.set_xlabel(r"Number of simulations ($\times 1000$)")
        ax.set_xticks([19, 39, 59, 79, 99])
        # ax.set_xticklabels([20000, 40000, 60000, 80000, 100000])
        ax.set_xticklabels([20, 40, 60, 80, 100])
        ax.set_ylim(ymin=0)
        ax.legend()
        plt.savefig(results_folder + "wass_dist_ABC_iterations.pdf", bbox_inches=bbox_inches)
        plt.close()

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
        plot_confidence_bands_performance_vs_iteration(RMSE_post_mean_ABC_FP, fig=fig, ax=ax, label="ABC-FP",
                                                       color_band_1="C1", color_line="C1", band_1=CI_level, alpha_1=0.3,
                                                       alpha_2=0, alpha_3=0, fill_between=True)
        plot_confidence_bands_performance_vs_iteration(RMSE_post_mean_ABC_SM, fig=fig, ax=ax, label="ABC-SM",
                                                       color_band_1="C2", color_line="C2", band_1=CI_level, alpha_1=0.3,
                                                       alpha_2=0, alpha_3=0, fill_between=True)
        # stack the following in order to have an horizontal plot:
        plot_confidence_bands_performance_vs_iteration(
            np.stack([RMSE_post_mean_exchange_SM] * max(ABC_FP_steps, ABC_SM_steps)), fig=fig, ax=ax,
            label="Exc-SM", color_band_1="blue", color_line="blue", band_1=CI_level, alpha_1=1, alpha_2=0,
            alpha_3=0, fill_between=False, ls="--", ls_band_1=":")
        ax.set_title(f"{model_text[model]}")
        ax.set_ylabel("RMSE posterior mean")
        # ax.set_xlabel("Number of simulations")
        ax.set_xlabel(r"Number of simulations ($\times 1000$)")
        ax.set_xticks([19, 39, 59, 79, 99])
        # ax.set_xticklabels([20000, 40000, 60000, 80000, 100000])
        ax.set_xticklabels([20, 40, 60, 80, 100])
        ax.set_ylim(ymin=0)
        ax.legend()
        plt.savefig(results_folder + "RMSE_post_mean_ABC_iterations.pdf", bbox_inches=bbox_inches)
        plt.close()

        iteration = check_iteration_better_perf(wass_dist_ABC_FP, wass_dist_exchange_SM)
        print("ABC-FP performs better than my method at iteration: ", None if iteration is None else iteration + 1)
        iteration = check_iteration_better_perf(wass_dist_ABC_SM, wass_dist_exchange_SM)
        print("ABC-SM performs better than my method at iteration: ", None if iteration is None else iteration + 1)

import os
import sys
from time import sleep, time

import numpy as np

# here we provide code for experiments with SL and RE.

sys.path.append(os.getcwd())  # for some reason it does not see my files if I don't put this

from abcpy.statistics import Identity
from abcpy.backends import BackendDummy, BackendMPI
from abcpy.continuousmodels import Uniform
from abcpy.inferences import PMC
from abcpy.approx_lhd import SynLikelihood, PenLogReg
from abcpy.output import Journal

from src.functions import subsample_trace, subsample_trace_and_weights, plot_single_marginal_with_trace_samples, \
    plot_bivariate_marginal_with_trace_samples, wass_dist, save_dict_to_json
from src.utils_arma_example import ARMAmodel, extract_params_and_weights_from_journal_ar2, \
    extract_params_and_weights_from_journal_ma2, extract_posterior_mean_from_journal_ar2, \
    extract_posterior_mean_from_journal_ma2, Autocorrelation
from src.utils_beta_example import IidBeta, BetaStatistics, \
    extract_params_and_weights_from_journal_beta, extract_posterior_mean_from_journal_beta
from src.utils_gaussian_example import IidNormal, GaussianStatistics, \
    extract_params_and_weights_from_journal_gaussian, extract_posterior_mean_from_journal_gaussian
from src.utils_gamma_example import IidGamma, GammaStatistics, \
    extract_params_and_weights_from_journal_gamma, extract_posterior_mean_from_journal_gamma
from src.parsers import parser_SL_RE

parser = parser_SL_RE()
args = parser.parse_args()

technique = args.technique
model = args.model
sleep_time = args.sleep
start_observation_index = args.start_observation_index
n_observations = args.n_observations
results_folder = args.root_folder
use_MPI = args.use_MPI
seed = args.seed
steps = args.steps
n_samples = args.n_samples
n_samples_per_param = args.n_samples_per_param
full_output = args.full_output
load_trace_if_available = args.load_trace_if_available
subsample_size = args.subsample_size
plot_marginal_densities = args.plot_marginal
plot_bivariate_densities = args.plot_bivariate
perform_postprocessing = args.postprocessing

if model not in ("gaussian", "beta", "gamma", "MA2", "AR2") or technique not in ("SL", "RE"):
    raise NotImplementedError

true_posterior_available = model in ("gaussian", "beta", "gamma", "MA2", "AR2")
theta_dim = 2

backend = BackendMPI() if use_MPI else BackendDummy()  # be careful, these need to be instantiated
print("{} model with {} approach.".format(model, technique))

args_dict = args.__dict__

if sleep_time > 0:
    print("Wait for {} minutes...".format(sleep_time))
    sleep(60 * sleep_time)
    print("Done waiting!")

if model == "AR2":
    arma_size = 100
    ar1_bounds = [-1, 1]
    ar2_bounds = [-1, 0]
    args_dict['arma_size'] = arma_size
    args_dict['ar1_bounds'] = ar1_bounds
    args_dict['ar2_bounds'] = ar2_bounds
    lower_bounds = np.array([ar1_bounds[0], ar2_bounds[0]])
    upper_bounds = np.array([ar1_bounds[1], ar2_bounds[1]])
    param_names = [r"$\theta_1$", r"$\theta_2$"]
    ar1 = Uniform([[ar1_bounds[0]], [ar1_bounds[1]]], name='ar1')
    ar2 = Uniform([[ar2_bounds[0]], [ar2_bounds[1]]], name='ar2')
    ABC_model = ARMAmodel([ar1, ar2], num_AR_params=2, num_MA_params=0, size=arma_size)
    statistic = Autocorrelation(2)
    if results_folder is None:
        results_folder = "results/AR2/"
    observation_folder = results_folder + '/' + args.observation_folder + "/"
    inference_folder = results_folder + '/' + args.inference_folder + "/"
    extract_params_and_weights_from_journal = extract_params_and_weights_from_journal_ar2
    extract_posterior_mean_from_journal = extract_posterior_mean_from_journal_ar2
elif model == "MA2":
    arma_size = 100
    ma1_bounds = [-1, 1]
    ma2_bounds = [0, 1]
    args_dict['arma_size'] = arma_size
    args_dict['ma1_bounds'] = ma1_bounds
    args_dict['ma2_bounds'] = ma2_bounds
    lower_bounds = np.array([ma1_bounds[0], ma2_bounds[0]])
    upper_bounds = np.array([ma1_bounds[1], ma2_bounds[1]])
    param_names = [r"$\theta_1$", r"$\theta_2$"]
    ma1 = Uniform([[ma1_bounds[0]], [ma1_bounds[1]]], name='ma1')
    ma2 = Uniform([[ma2_bounds[0]], [ma2_bounds[1]]], name='ma2')
    ABC_model = ARMAmodel([ma1, ma2], num_AR_params=0, num_MA_params=2, size=arma_size)
    statistic = Autocorrelation(2)
    if results_folder is None:
        results_folder = "results/MA2/"
    observation_folder = results_folder + '/' + args.observation_folder + "/"
    inference_folder = results_folder + '/' + args.inference_folder + "/"
    extract_params_and_weights_from_journal = extract_params_and_weights_from_journal_ma2
    extract_posterior_mean_from_journal = extract_posterior_mean_from_journal_ma2
elif model == "beta":
    alpha_bounds = [1, 3]
    beta_bounds = [1, 3]
    args_dict['alpha_bounds'] = alpha_bounds
    args_dict['beta_bounds'] = beta_bounds
    lower_bounds = np.array([alpha_bounds[0], beta_bounds[0]])
    upper_bounds = np.array([alpha_bounds[1], beta_bounds[1]])
    param_names = [r"$\alpha$", r"$\beta$"]
    alpha_abc = Uniform([[alpha_bounds[0]], [alpha_bounds[1]]], name='alpha')
    beta_param_abc = Uniform([[beta_bounds[0]], [beta_bounds[1]]], name='beta')
    ABC_model = IidBeta([alpha_abc, beta_param_abc], iid_size=10, name='beta_final_model')
    statistic = BetaStatistics()
    if results_folder is None:
        results_folder = "results/beta/"
    observation_folder = results_folder + '/' + args.observation_folder + "/"
    inference_folder = results_folder + '/' + args.inference_folder + "/"
    extract_params_and_weights_from_journal = extract_params_and_weights_from_journal_beta
    extract_posterior_mean_from_journal = extract_posterior_mean_from_journal_beta
elif model == "gamma":
    k_bounds = [1, 3]  # this is what I used in the likelihood models setup.
    theta_bounds = [1, 3]
    # k_bounds = [0.1, 1]
    # theta_bounds = [0, 1]
    args_dict['k_bounds'] = k_bounds
    args_dict['theta_bounds'] = theta_bounds
    lower_bounds = np.array([k_bounds[0], theta_bounds[0]])
    upper_bounds = np.array([k_bounds[1], theta_bounds[1]])
    param_names = [r"$k$", r"$\theta$"]
    k_abc = Uniform([[k_bounds[0]], [k_bounds[1]]], name='k')
    theta_abc = Uniform([[theta_bounds[0]], [theta_bounds[1]]], name='theta')
    ABC_model = IidGamma([k_abc, theta_abc], iid_size=10, name='gamma')
    statistic = GammaStatistics()
    if results_folder is None:
        results_folder = "results/gamma/"
    observation_folder = results_folder + '/' + args.observation_folder + "/"
    inference_folder = results_folder + '/' + args.inference_folder + "/"
    extract_params_and_weights_from_journal = extract_params_and_weights_from_journal_gamma
    extract_posterior_mean_from_journal = extract_posterior_mean_from_journal_gamma
elif model == "gaussian":
    mu_bounds = [-10, 10]
    sigma_bounds = [1, 10]
    args_dict['mu_bounds'] = mu_bounds
    args_dict['sigma_bounds'] = sigma_bounds
    lower_bounds = np.array([mu_bounds[0], sigma_bounds[0]])
    upper_bounds = np.array([mu_bounds[1], sigma_bounds[1]])
    param_names = [r"$\mu$", r"$\sigma$"]
    mu_abc = Uniform([[mu_bounds[0]], [mu_bounds[1]]], name='mu')
    sigma_abc = Uniform([[sigma_bounds[0]], [sigma_bounds[1]]], name='sigma')
    ABC_model = IidNormal([mu_abc, sigma_abc], iid_size=10, name='gaussian')
    statistic = GaussianStatistics()
    if results_folder is None:
        results_folder = "results/gaussian/"
    observation_folder = results_folder + '/' + args.observation_folder + "/"
    inference_folder = results_folder + '/' + args.inference_folder + "/"
    extract_params_and_weights_from_journal = extract_params_and_weights_from_journal_gaussian
    extract_posterior_mean_from_journal = extract_posterior_mean_from_journal_gaussian
else:
    raise NotImplementedError
save_dict_to_json(args.__dict__, inference_folder + 'config.json')

# now setup the Synthetic likelihood experiments or ratio estimation one:
if technique == "SL":
    approx_lhd = SynLikelihood(statistic)
elif technique == "RE":
    # for the RE approach: it is better to use pairwise combinations of the statistics in order to make comparison with
    # SL fair
    statistic = Identity(cross=True, previous_statistics=statistic,
                         degree=1)  # this should automatically use the pairwise comb.
    # when instantiating this, it takes additional parameters; does it simulate from the model immediately?
    approx_lhd = PenLogReg(statistic, [ABC_model], n_samples_per_param, n_folds=10, max_iter=100000, seed=seed)
else:
    raise NotImplementedError

sampler = PMC([ABC_model], [approx_lhd], backend, seed=seed)

wass_dist_array = np.zeros(n_observations)
RMSE_post_mean = np.zeros(n_observations)
running_time_inference = np.zeros(n_observations)

namefile_postfix_no_index = "_steps_{}_n_samples_{}_n_samples_per_param_{}".format(steps, n_samples,
                                                                                   n_samples_per_param)
if perform_postprocessing:
    theta_test = np.zeros((n_observations - start_observation_index, theta_dim))
    traces = np.zeros((n_samples, n_observations - start_observation_index, theta_dim))

for obs_index in range(start_observation_index, n_observations):
    print("Observation {}".format(obs_index + 1))

    namefile_postfix = "_{}_steps_{}_n_samples_{}_n_samples_per_param_{}".format(obs_index + 1, steps, n_samples,
                                                                                 n_samples_per_param)

    # load obs
    x_obs = [np.load(observation_folder + "x_obs{}.npy".format(obs_index + 1))]
    theta_obs = np.load(observation_folder + "theta_obs{}.npy".format(obs_index + 1))
    if true_posterior_available:
        trace_true = np.load(observation_folder + "true_mcmc_trace{}.npy".format(obs_index + 1))
        trace_true_subsample = subsample_trace(trace_true,
                                               size=subsample_size)  # used to compute wass dist and for bivariate plot
        true_post_means = np.mean(trace_true, axis=0)

    # check the statistic computation:
    print(statistic.statistics(x_obs))

    if load_trace_if_available:
        try:
            jrnl = Journal.fromFile(inference_folder + "jrnl" + namefile_postfix + ".jnl")
            perform_ABC = False
            print("\n Using previosly generated journal.")
        except FileNotFoundError:
            perform_ABC = True
    else:
        perform_ABC = True

    if perform_ABC:
        print("Performing inference...")
        start = time()
        # steps, n_samples = 10000, n_samples_per_param = 100
        start = time()
        jrnl = sampler.sample([x_obs], steps, n_samples=n_samples, n_samples_per_param=n_samples_per_param,
                              full_output=1 if full_output else 0)
        end = time()
        print("It took {:.4f} seconds".format(end - start))
        running_time_inference[obs_index] = end - start
        np.save(inference_folder + "running_time_inference" + namefile_postfix_no_index, running_time_inference)
        jrnl.save(inference_folder + "jrnl" + namefile_postfix + ".jnl")

    params, weights = extract_params_and_weights_from_journal(jrnl)
    means = extract_posterior_mean_from_journal(jrnl)
    params_subsample, weights_subsample = subsample_trace_and_weights(params, weights, size=subsample_size)
    if perform_postprocessing:
        traces[:, obs_index - start_observation_index, :] = params  # store this
        theta_test[obs_index - start_observation_index] = theta_obs

    if true_posterior_available:
        # wass dist and RMSE:
        wass_dist_array[obs_index] = wass_dist(params_subsample, trace_true_subsample,
                                               weights_post_1=weights_subsample)
        RMSE_post_mean[obs_index] = np.linalg.norm(means - true_post_means)

        np.save(inference_folder + "wass_dist" + namefile_postfix_no_index, wass_dist_array)
        np.save(inference_folder + "RMSE_post_mean" + namefile_postfix_no_index, RMSE_post_mean)

    if plot_marginal_densities:
        plot_single_marginal_with_trace_samples(theta_obs, params, trace_true if true_posterior_available else None,
                                                weights_trace_approx=weights, param_names=param_names,
                                                namefile=inference_folder + "single_marginals" + namefile_postfix + ".png", )

    if plot_bivariate_densities:
        plot_bivariate_marginal_with_trace_samples(
            theta_obs, params, trace_true if true_posterior_available else None,
            weights_trace_approx=weights,
            thetarange=np.array([lower_bounds, upper_bounds]),
            namefile=inference_folder + "bivariate_marginals" + namefile_postfix + ".png", color="#40739E",
            figsize_vertical=5, title_size=16, label_size=16, param1_name=param_names[0],
            param2_name=param_names[1])

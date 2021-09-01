import logging
import os
import pickle
import sys
from time import sleep

import torch

sys.path.append(os.getcwd())
import os
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

sys.path.append(os.getcwd())

from src.functions import determine_eps, wass_dist, subsample_trace, ABC_inference, subsample_trace_and_weights, \
    RescaleAndNet, RescaleAndDiscardLastOutputNet, plot_single_marginal_with_trace_samples, \
    plot_bivariate_marginal_with_trace_samples, plot_trace, save_dict_to_json, generate_training_samples_ABC_model, \
    DummyScaler
from src.exchange_mcmc import exchange_MCMC_with_SM_statistics, uniform_prior_theta
from src.networks import createDefaultNN, createDefaultNNWithDerivatives, create_PEN_architecture
from src.utils_arma_example import ARMAmodel, extract_params_and_weights_from_journal_ar2, \
    extract_params_and_weights_from_journal_ma2, extract_posterior_mean_from_journal_ar2, \
    extract_posterior_mean_from_journal_ma2
from src.utils_beta_example import TrueSummariesComputationBeta, IidBeta, \
    extract_params_and_weights_from_journal_beta, extract_posterior_mean_from_journal_beta, \
    generate_beta_training_samples
from src.utils_gaussian_example import TrueSummariesComputationGaussian, IidNormal, \
    extract_params_and_weights_from_journal_gaussian, extract_posterior_mean_from_journal_gaussian, \
    generate_gaussian_training_samples
from src.utils_gamma_example import TrueSummariesComputationGamma, IidGamma, \
    extract_params_and_weights_from_journal_gamma, extract_posterior_mean_from_journal_gamma, \
    generate_gamma_training_samples
from src.utils_Lorenz95_example import extract_params_and_weights_from_journal_Lorenz95, \
    extract_posterior_mean_from_journal_Lorenz95, StochLorenz95_with_statistics
from src.parsers import parser_approx_likelihood_approach

from abcpy.continuousmodels import Uniform
from abcpy.NN_utilities.utilities import load_net
from abcpy.statistics import NeuralEmbedding
from abcpy.distances import Euclidean
from abcpy.backends import BackendMPI, BackendDummy
from src.functions import WeightedEuclidean
from abcpy.output import Journal

parser = parser_approx_likelihood_approach(None, default_nets_folder="net-SM", default_inference_folder="Exc-SM",
                                           default_n_samples=1000, default_burnin_MCMC=5000)

args = parser.parse_args()
technique = args.technique
model = args.model
inference_technique = args.inference_technique  # this defaults to exchange
sleep_time = args.sleep
start_observation_index = args.start_observation_index
n_observations = args.n_observations
n_samples = args.n_samples
burnin_exchange_MCMC = args.burnin_MCMC
aux_MCMC_inner_steps_exchange_MCMC = args.aux_MCMC_inner_steps_exchange_MCMC
aux_MCMC_proposal_size_exchange_MCMC = args.aux_MCMC_proposal_size_exchange_MCMC
load_trace_if_available = args.load_trace_if_available
ABC_algorithm = args.ABC_alg
ABC_steps = args.ABC_steps
ABC_full_output = args.ABC_full_output
ABC_eps = args.ABC_eps
weighted_euclidean_distance = not args.no_weighted_eucl_dist
SMCABC_use_robust_kernel = args.SMCABC_use_robust_kernel
SABC_cutoff = args.SABC_cutoff
results_folder = args.root_folder
nets_folder = args.nets_folder
observation_folder = args.observation_folder
inference_folder = args.inference_folder
batch_norm_last_layer = not args.no_bn
affine_batch_norm = args.affine_bn
plot_marginal_densities = args.plot_marginal
plot_bivariate_densities = args.plot_bivariate
plot_trace_flag = args.plot_trace
plot_journal = args.plot_journal
subsample_size = args.subsample_size
use_MPI = args.use_MPI
epsilon_quantile = args.epsilon_quantile
seed = args.seed
debug_level = logging.INFO if args.debug_level == "info" else logging.DEBUG if args.debug_level == "debug" else \
    logging.WARN if args.debug_level == "warn" else logging.ERROR if args.debug_level == "error" else logging.WARN
tuning_window_exchange_MCMC = args.tuning_window_exchange_MCMC
propose_new_theta_exchange_MCMC = args.propose_new_theta_exchange_MCMC
bridging_exch_MCMC = args.bridging_exch_MCMC

np.random.seed(seed)

# checks
if model not in ("gaussian", "beta", "gamma", "MA2", "AR2", "Lorenz95") or technique not in ("SM", "SSM", "FP"):
    raise NotImplementedError

true_posterior_available = model not in ("Lorenz95",)

if inference_technique not in ("exchange", "ABC"):
    raise NotImplementedError

print("{} model with {}.".format(model, technique))
# set up the default root folder and other values
default_root_folder = {"gaussian": "results/gaussian/",
                       "gamma": "results/gamma/",
                       "beta": "results/beta/",
                       "AR2": "results/AR2/",
                       "MA2": "results/MA2/",
                       "Lorenz95": "results/Lorenz95/"}
if results_folder is None:
    results_folder = default_root_folder[model]
if nets_folder is None:
    nets_folder = "net-" + technique if technique in ["SM", "SSM"] else "net-FP"

results_folder = results_folder + '/'
nets_folder = results_folder + nets_folder + '/'
observation_folder = results_folder + observation_folder + '/'
inference_folder = results_folder + inference_folder + '/'

if sleep_time > 0:
    print("Wait for {} minutes...".format(sleep_time))
    sleep(60 * sleep_time)
    print("Done waiting!")

save_exchange_MCMC_trace = True
save_jrnl = True

args_dict = args.__dict__
# add other arguments to the config dicts
args_dict['save_exchange_MCMC_trace'] = save_exchange_MCMC_trace
args_dict['save_jrnl'] = save_jrnl

if model == "gaussian":
    mu_bounds = [-10, 10]
    sigma_bounds = [1, 10]
    args_dict['mu_bounds'] = mu_bounds
    args_dict['sigma_bounds'] = sigma_bounds
    lower_bounds = np.array([mu_bounds[0], sigma_bounds[0]])
    upper_bounds = np.array([mu_bounds[1], sigma_bounds[1]])
    initial_theta_exchange_MCMC = np.array([0, 5.5])
    proposal_size_exchange_MCMC = 2 * np.array([1, .5])
    theta_dim = 2
    param_names = [r"$\mu$", r"$\sigma$"]

    # generate some data for defining the weighted distances and the epsilon:
    theta_vect_test, samples_matrix_test = generate_gaussian_training_samples(n_theta=1000,
                                                                              size_iid_samples=10, mu_bounds=mu_bounds,
                                                                              sigma_bounds=sigma_bounds)

    # define ABC graphical model:
    mu_abc = Uniform([[mu_bounds[0]], [mu_bounds[1]]], name='mu')
    sigma_abc = Uniform([[sigma_bounds[0]], [sigma_bounds[1]]], name='sigma')
    ABC_model = IidNormal([mu_abc, sigma_abc], iid_size=10, name='gaussian')

    # define the functions that are needed:
    TrueSummariesComputation = TrueSummariesComputationGaussian
    extract_params_and_weights_from_journal = extract_params_and_weights_from_journal_gaussian
    extract_posterior_mean_from_journal = extract_posterior_mean_from_journal_gaussian

elif model == "beta":
    alpha_bounds = [1, 3]
    beta_bounds = [1, 3]
    args_dict['alpha_bounds'] = alpha_bounds
    args_dict['beta_bounds'] = beta_bounds
    lower_bounds = np.array([alpha_bounds[0], beta_bounds[0]])
    upper_bounds = np.array([alpha_bounds[1], beta_bounds[1]])
    initial_theta_exchange_MCMC = np.array([2, 2])
    if propose_new_theta_exchange_MCMC in ("transformation", "adaptive_transformation"):
        proposal_size_exchange_MCMC = np.ones_like(initial_theta_exchange_MCMC, dtype=float)
    else:
        proposal_size_exchange_MCMC = np.array([0.2, 0.2])
    theta_dim = 2
    param_names = [r"$\alpha$", r"$\beta$"]

    # generate some data for defining the weighted distances and the epsilon:
    theta_vect_test, samples_matrix_test = generate_beta_training_samples(n_theta=1000,
                                                                          size_iid_samples=10,
                                                                          alpha_bounds=alpha_bounds,
                                                                          beta_bounds=beta_bounds)

    # define ABC graphical model:
    alpha_abc = Uniform([[alpha_bounds[0]], [alpha_bounds[1]]], name="alpha")
    beta_param_abc = Uniform([[beta_bounds[0]], [beta_bounds[1]]], name="beta")
    ABC_model = IidBeta([alpha_abc, beta_param_abc], iid_size=10, name="beta_final_model")

    TrueSummariesComputation = TrueSummariesComputationBeta
    extract_params_and_weights_from_journal = extract_params_and_weights_from_journal_beta
    extract_posterior_mean_from_journal = extract_posterior_mean_from_journal_beta

elif model == "gamma":
    k_bounds = [1, 3]
    theta_bounds = [1, 3]
    args_dict['k_bounds'] = k_bounds
    args_dict['theta_bounds'] = theta_bounds
    lower_bounds = np.array([k_bounds[0], theta_bounds[0]])
    upper_bounds = np.array([k_bounds[1], theta_bounds[1]])
    initial_theta_exchange_MCMC = np.array([2, 2])
    proposal_size_exchange_MCMC = np.array([0.2, 0.2])
    theta_dim = 2
    param_names = [r"$k$", r"$\theta$"]

    # generate some data for defining the weighted distances and the epsilon:
    theta_vect_test, samples_matrix_test = generate_gamma_training_samples(n_theta=1000,
                                                                           size_iid_samples=10, k_bounds=k_bounds,
                                                                           theta_bounds=theta_bounds)

    # define ABC graphical model:
    k_abc = Uniform([[k_bounds[0]], [k_bounds[1]]], name='k')
    theta_abc = Uniform([[theta_bounds[0]], [theta_bounds[1]]], name='theta')
    ABC_model = IidGamma([k_abc, theta_abc], iid_size=10, name='gamma')

    TrueSummariesComputation = TrueSummariesComputationGamma
    extract_params_and_weights_from_journal = extract_params_and_weights_from_journal_gamma
    extract_posterior_mean_from_journal = extract_posterior_mean_from_journal_gamma

elif model == "AR2":
    arma_size = 100
    ar1_bounds = [-1, 1]
    ar2_bounds = [-1, 0]
    args_dict['arma_size'] = arma_size
    args_dict['ar1_bounds'] = ar1_bounds
    args_dict['ar2_bounds'] = ar2_bounds
    lower_bounds = np.array([ar1_bounds[0], ar2_bounds[0]])
    upper_bounds = np.array([ar1_bounds[1], ar2_bounds[1]])
    initial_theta_exchange_MCMC = np.array([0, -0.5])
    proposal_size_exchange_MCMC = np.array([0.2, 0.1])
    theta_dim = 2
    param_names = [r"$\theta_1$", r"$\theta_2$"]

    ar1 = Uniform([[ar1_bounds[0]], [ar1_bounds[1]]], name='ar1')
    ar2 = Uniform([[ar2_bounds[0]], [ar2_bounds[1]]], name='ar2')
    ABC_model = ARMAmodel([ar1, ar2], num_AR_params=2, num_MA_params=0, size=arma_size)

    theta_vect_test, samples_matrix_test = generate_training_samples_ABC_model(ABC_model, 1000, seed=seed)
    # temporary fix, as the inference code expects this to be a pytorch tensor. Need to organize better the way
    # torch/numpy is used here.
    samples_matrix_test = torch.tensor(samples_matrix_test, requires_grad=False, dtype=torch.float)

    TrueSummariesComputation = None
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
    initial_theta_exchange_MCMC = np.array([0, 0.5])
    if propose_new_theta_exchange_MCMC in ("transformation", "adaptive_transformation"):
        proposal_size_exchange_MCMC = np.ones_like(initial_theta_exchange_MCMC, dtype=float) * 0.4
    else:
        proposal_size_exchange_MCMC = np.array([0.2, 0.1])
    theta_dim = 2
    param_names = [r"$\theta_1$", r"$\theta_2$"]

    ma1 = Uniform([[ma1_bounds[0]], [ma1_bounds[1]]], name='ma1')
    ma2 = Uniform([[ma2_bounds[0]], [ma2_bounds[1]]], name='ma2')
    ABC_model = ARMAmodel([ma1, ma2], num_AR_params=0, num_MA_params=2, size=arma_size)

    theta_vect_test, samples_matrix_test = generate_training_samples_ABC_model(ABC_model, 1000, seed=seed)
    samples_matrix_test = torch.tensor(samples_matrix_test, requires_grad=False, dtype=torch.float)

    TrueSummariesComputation = None
    extract_params_and_weights_from_journal = extract_params_and_weights_from_journal_ma2
    extract_posterior_mean_from_journal = extract_posterior_mean_from_journal_ma2

elif model == "Lorenz95":
    # we follow here the same setup as in the ratio estimation paper. Then we only infer theta_1, theta_2 and keep
    # fixed sigma_3 and phi. We moreover change prior range and use the Hakkarainen statistics

    theta1_min = 1.4
    theta1_max = 2.2
    theta2_min = 0
    theta2_max = 1
    sigma_e_min = 1.5
    sigma_e_max = 2.5
    phi_min = 0
    phi_max = 1

    lower_bounds = np.array([theta1_min, theta2_min, sigma_e_min, phi_min])
    upper_bounds = np.array([theta1_max, theta2_max, sigma_e_max, phi_max])
    initial_theta_exchange_MCMC = np.array([1.8, 0.5, 2.0, 0.5])
    # proposal_size_exchange_MCMC = np.array([0.05, 0.05, 0.05, 0.05])
    if propose_new_theta_exchange_MCMC in ("transformation", "adaptive_transformation"):
        proposal_size_exchange_MCMC = np.ones_like(initial_theta_exchange_MCMC, dtype=float)
    else:
        proposal_size_exchange_MCMC = (upper_bounds - lower_bounds) / 15
    theta_dim = 4
    param_names = [r"$\theta_1$", r"$\theta_2$", r"$\sigma_e$", r"$\phi$"]

    theta1 = Uniform([[theta1_min], [theta1_max]], name='theta1')
    theta2 = Uniform([[theta2_min], [theta2_max]], name='theta2')
    # sigma_e = Exponential([[sigma_e_rate]], name='sigma_e')
    sigma_e = Uniform([[sigma_e_min], [sigma_e_max]], name='sigma_e')
    phi = Uniform([[phi_min], [phi_max]], name='phi')
    ABC_model = StochLorenz95_with_statistics([theta1, theta2, sigma_e, phi], time_units=4,
                                              n_timestep_per_time_unit=30, name='lorenz')

    print("Generate test data:")
    if inference_technique == "ABC":
        theta_vect_test, samples_matrix_test = generate_training_samples_ABC_model(ABC_model, 1000, seed=seed)
        samples_matrix_test = torch.tensor(samples_matrix_test, requires_grad=False, dtype=torch.float)
    else:
        theta_vect_test, samples_matrix_test = None, None
    # compute statistics here as well:

    TrueSummariesComputation = None
    extract_params_and_weights_from_journal = extract_params_and_weights_from_journal_Lorenz95
    extract_posterior_mean_from_journal = extract_posterior_mean_from_journal_Lorenz95

args_dict['initial_theta_exchange_MCMC'] = initial_theta_exchange_MCMC.tolist()
args_dict['proposal_size_exchange_MCMC'] = proposal_size_exchange_MCMC.tolist()
save_dict_to_json(args.__dict__, inference_folder + 'config.json')

if model in ("beta", "gamma", "gaussian"):
    # define network architectures:
    nonlinearity = torch.nn.Softplus
    # nonlinearity = torch.nn.Tanhshrink
    # net_data_SM_architecture = createDefaultNN(10, 3, [30, 50, 50, 20], nonlinearity=nonlinearity())
    net_data_SM_architecture = createDefaultNNWithDerivatives(10, 3, [30, 50, 50, 20], nonlinearity=nonlinearity)
    # net_data_SM_architecture = createDefaultNN(10, 3, [15, 15, 5], nonlinearity=nonlinearity())
    # net_theta_SM_architecture = createDefaultNN(2, 2, [5, 5], nonlinearity=nonlinearity())
    net_theta_SM_architecture = createDefaultNN(2, 2, [15, 30, 30, 15], nonlinearity=nonlinearity(),
                                                batch_norm_last_layer=batch_norm_last_layer,
                                                affine_batch_norm=affine_batch_norm)
    # net_FP_architecture = createDefaultNN(10, 2, [30, 50, 50, 20], nonlinearity=nonlinearity())
    net_FP_architecture = createDefaultNN(10, 2, [30, 50, 50, 20], nonlinearity=torch.nn.ReLU())
    # net_FP_architecture = createDefaultNN(10, 2, [15, 15, 5], nonlinearity=nonlinearity())

elif model == "MA2":
    nonlinearity = torch.nn.Softplus

    # the batchnorm has to be put here in the theta net.
    # PEN 2:
    # phi_net_data_architecture = createDefaultNNWithDerivatives(3, 20, [50, 50, 30], nonlinearity=nonlinearity)
    # rho_net_data_architecture = createDefaultNNWithDerivatives(22, 3, [50, 50], nonlinearity=nonlinearity)
    # net_data_SM_architecture = create_PEN_architecture(phi_net_data_architecture, rho_net_data_architecture, 2)
    # phi_net_FP_architecture = createDefaultNN(3, 20, [50, 50, 30], nonlinearity=nonlinearity())
    # rho_net_FP_architecture = createDefaultNN(22, 2, [50, 50], nonlinearity=nonlinearity())
    # net_FP_architecture = create_PEN_architecture(phi_net_FP_architecture, rho_net_FP_architecture, 2)
    # PEN 10:
    phi_net_data_architecture = createDefaultNNWithDerivatives(11, 20, [50, 50, 30], nonlinearity=nonlinearity)
    rho_net_data_architecture = createDefaultNNWithDerivatives(30, 3, [50, 50], nonlinearity=nonlinearity)
    net_data_SM_architecture = create_PEN_architecture(phi_net_data_architecture, rho_net_data_architecture, 10)

    phi_net_FP_architecture = createDefaultNN(11, 20, [50, 50, 30], nonlinearity=nonlinearity())
    rho_net_FP_architecture = createDefaultNN(30, 2, [50, 50], nonlinearity=nonlinearity())
    net_FP_architecture = create_PEN_architecture(phi_net_FP_architecture, rho_net_FP_architecture, 10)

    # net_energy_architecture = createDefaultNN(4, 1, [15, 30, 30, 15], nonlinearity=nonlinearity(),
    #                                           batch_norm_last_layer=batch_norm_last_layer,
    #                                           affine_batch_norm=affine_batch_norm)
    net_theta_SM_architecture = createDefaultNN(2, 2, [15, 30, 30, 15], nonlinearity=nonlinearity(),
                                                batch_norm_last_layer=batch_norm_last_layer,
                                                affine_batch_norm=affine_batch_norm)

elif model == "AR2":
    nonlinearity = torch.nn.Softplus
    phi_net_data_architecture = createDefaultNNWithDerivatives(3, 20, [50, 50, 30], nonlinearity=nonlinearity)
    rho_net_data_architecture = createDefaultNNWithDerivatives(22, 3, [50, 50], nonlinearity=nonlinearity)
    net_data_SM_architecture = create_PEN_architecture(phi_net_data_architecture, rho_net_data_architecture, 2)

    phi_net_FP_architecture = createDefaultNN(3, 20, [50, 50, 30], nonlinearity=nonlinearity())
    rho_net_FP_architecture = createDefaultNN(22, 2, [50, 50], nonlinearity=nonlinearity())
    net_FP_architecture = create_PEN_architecture(phi_net_FP_architecture, rho_net_FP_architecture, 2)

    # net_energy_architecture = createDefaultNN(4, 1, [15, 30, 30, 15], nonlinearity=nonlinearity(),
    #                                           batch_norm_last_layer=batch_norm_last_layer,
    #                                           affine_batch_norm=affine_batch_norm)
    net_theta_SM_architecture = createDefaultNN(2, 2, [15, 30, 30, 15], nonlinearity=nonlinearity(),
                                                batch_norm_last_layer=batch_norm_last_layer,
                                                affine_batch_norm=affine_batch_norm)

elif model == "Lorenz95":
    # define network architectures:
    nonlinearity = torch.nn.Softplus
    net_data_SM_architecture = createDefaultNNWithDerivatives(23, 5, [70, 120, 120, 70, 20], nonlinearity=nonlinearity)
    net_theta_SM_architecture = createDefaultNN(4, 4, [30, 50, 50, 30], nonlinearity=nonlinearity(),
                                                batch_norm_last_layer=batch_norm_last_layer,
                                                affine_batch_norm=affine_batch_norm)
    net_FP_architecture = createDefaultNN(23, 4, [70, 120, 120, 70, 20],
                                          nonlinearity=torch.nn.ReLU())  # I am using relu here

# load nets
if technique in ["SM", "SSM"]:
    net_theta_SM = load_net(nets_folder + "net_theta_SM.pth", net_theta_SM_architecture).eval()
    net_data_SM = load_net(nets_folder + "net_data_SM.pth", net_data_SM_architecture).eval()
    scaler_data_SM = pickle.load(open(nets_folder + "scaler_data_SM.pkl", "rb"))
    scaler_theta_SM = pickle.load(open(nets_folder + "scaler_theta_SM.pkl", "rb"))
    if scaler_data_SM is None:
        scaler_data_SM = DummyScaler()
    if scaler_theta_SM is None:
        scaler_theta_SM = DummyScaler()
elif technique == "FP":
    net_FP = load_net(nets_folder + "net_data_FP.pth", net_FP_architecture)
    scaler_data_FP = pickle.load(open(nets_folder + "scaler_data_FP.pkl", "rb"))
else:
    raise NotImplementedError

# ABC inference kwargs
kwargs_abc_inference = {}
if ABC_algorithm == "SMCABC":
    kwargs_abc_inference['which_mcmc_kernel'] = 1 if SMCABC_use_robust_kernel else 0
elif ABC_algorithm == "SABC":
    kwargs_abc_inference['ar_cutoff'] = SABC_cutoff

if inference_technique == "ABC":
    namefile_postfix_no_index = "_{}_{}_steps_{}_n_samples_{}".format(ABC_algorithm, technique,
                                                                      ABC_steps, n_samples)
elif inference_technique == "exchange":
    namefile_postfix_no_index = "_{}_n_samples_{}".format("exchange", n_samples)

wass_dist_ABC = np.zeros(n_observations)
wass_dist_exchange_SM = np.zeros(n_observations)

RMSE_post_mean_ABC = np.zeros(n_observations)
RMSE_post_mean_exchange_SM = np.zeros(n_observations)

try:
    # load running time so that we do not lose it when adding additional observations.
    running_time_inference_ABC = np.load(
        inference_folder + "running_time_inference" + namefile_postfix_no_index + ".npy")
    if running_time_inference_ABC.shape[0] < n_observations:
        running_time_inference_ABC = np.concatenate(
            (running_time_inference_ABC, np.zeros(n_observations - running_time_inference_ABC.shape[0])))
    running_time_inference_exchange_SM = np.copy(running_time_inference_ABC)

except FileNotFoundError:
    running_time_inference_ABC = np.zeros(n_observations)
    running_time_inference_exchange_SM = np.zeros(n_observations)

# Define backend
backend = BackendMPI() if use_MPI else BackendDummy()

for obs_index in range(start_observation_index, n_observations):
    print("Observation ", obs_index + 1)
    namefile_postfix = "_{}".format(obs_index + 1) + namefile_postfix_no_index
    x_obs = np.load(observation_folder + "x_obs{}.npy".format(obs_index + 1))
    theta_obs = np.load(observation_folder + "theta_obs{}.npy".format(obs_index + 1))
    if true_posterior_available:
        trace_true = np.load(observation_folder + "true_mcmc_trace{}.npy".format(obs_index + 1))
        trace_true_subsample = subsample_trace(trace_true, size=min(subsample_size, n_samples))  # otherwise it
        # crashes if subsample_size>n_samples
        true_post_means = np.mean(trace_true, axis=0)

    # INFERENCE STEP

    if inference_technique == "exchange" and technique in ["SM", "SSM"]:
        print(f"\nPerforming exchange MCMC inference with {technique}.")
        if load_trace_if_available:
            try:
                trace_exchange_burned_in = np.load(
                    inference_folder + "exchange_mcmc_trace{}.npy".format(obs_index + 1))
                generate_new_MCMC = False
                print("\n Using previosly generated approx posterior.")
            except FileNotFoundError:
                generate_new_MCMC = True
        else:
            generate_new_MCMC = True

        if generate_new_MCMC:
            start = time()
            trace_exchange = exchange_MCMC_with_SM_statistics(x_obs, initial_theta_exchange_MCMC,
                                                              lambda x: uniform_prior_theta(x, lower_bounds,
                                                                                            upper_bounds),
                                                              net_data_SM, net_theta_SM, scaler_data_SM,
                                                              scaler_theta_SM, propose_new_theta_exchange_MCMC,
                                                              T=n_samples, burn_in=burnin_exchange_MCMC,
                                                              tuning_window_size=tuning_window_exchange_MCMC,
                                                              aux_MCMC_inner_steps=aux_MCMC_inner_steps_exchange_MCMC,
                                                              aux_MCMC_proposal_size=aux_MCMC_proposal_size_exchange_MCMC,
                                                              K=bridging_exch_MCMC,
                                                              seed=seed,
                                                              debug_level=debug_level,
                                                              lower_bounds_theta=lower_bounds,
                                                              upper_bounds_theta=upper_bounds,
                                                              sigma=proposal_size_exchange_MCMC)

            trace_exchange_burned_in = trace_exchange[burnin_exchange_MCMC:]
            running_time_inference_exchange_SM[obs_index] = time() - start
            np.save(inference_folder + "running_time_inference" + namefile_postfix_no_index,
                    running_time_inference_exchange_SM)

            if save_exchange_MCMC_trace:
                np.save(inference_folder + "exchange_mcmc_trace{}".format(obs_index + 1), trace_exchange_burned_in)

        means_exchange = np.mean(trace_exchange_burned_in, axis=0)
        trace_exchange_subsample = subsample_trace(trace_exchange_burned_in,
                                                   size=subsample_size)  # used to compute wass dist
        # compute wass distance and RMSE
        if true_posterior_available:
            print(f"Wass dist: {wass_dist(trace_exchange_subsample, trace_true_subsample):.4f}")
            wass_dist_exchange_SM[obs_index] = wass_dist(trace_exchange_subsample, trace_true_subsample)
            RMSE_post_mean_exchange_SM[obs_index] = np.linalg.norm(means_exchange - true_post_means)

            # this does not agree to the comparison one
            np.save(inference_folder + "wass_dist" + namefile_postfix_no_index, wass_dist_exchange_SM)
            np.save(inference_folder + "RMSE_post_mean" + namefile_postfix_no_index, RMSE_post_mean_exchange_SM)

        # alias:
        params = trace_exchange_burned_in
        params_subsample = trace_exchange_subsample

    elif "ABC" == inference_technique:

        if load_trace_if_available:
            try:
                jrnl = Journal.fromFile(inference_folder + "jrnl" + namefile_postfix + ".jnl")
                perform_ABC = False
                print("\n Using previosly generated approx posterior.")
            except FileNotFoundError:
                perform_ABC = True
        else:
            perform_ABC = True

        # you can perform ABC inference with both SM, FP statistics and the true ones
        if technique in ["SM", "SSM"]:
            if perform_ABC:
                print(f"\nPerform ABC inference with {technique} statistics.")
                if weighted_euclidean_distance:  # define the distance object
                    # keep the last 100 test samples to estimate the initial eps value if not provided
                    distance_calculator = WeightedEuclidean(
                        NeuralEmbedding(RescaleAndDiscardLastOutputNet(net_data_SM, scaler_data_SM)),
                        [samples_matrix_test[i].numpy() for i in
                         range(samples_matrix_test.shape[0] - 100 * (ABC_eps is None))])
                else:
                    distance_calculator = Euclidean(
                        NeuralEmbedding(RescaleAndDiscardLastOutputNet(net_data_SM, scaler_data_SM)))

        elif "FP" == technique:
            if perform_ABC:
                print("\nPerform ABC inference with FP statistics.")
                if weighted_euclidean_distance:
                    distance_calculator = WeightedEuclidean(
                        NeuralEmbedding(RescaleAndNet(net_FP, scaler_data_FP)),
                        [samples_matrix_test[i].numpy() for i in
                         range(samples_matrix_test.shape[0] - 100 * (ABC_eps is None))])
                else:
                    distance_calculator = Euclidean(NeuralEmbedding(RescaleAndNet(net_FP, scaler_data_FP)))

        elif "true" == technique:
            if perform_ABC:
                print("\nPerform ABC inference with true statistics.")
                if weighted_euclidean_distance:
                    distance_calculator = WeightedEuclidean(NeuralEmbedding(TrueSummariesComputation()),
                                                            [samples_matrix_test[i].numpy() for i in
                                                             range(samples_matrix_test.shape[0] - 100 * (
                                                                     ABC_eps is None))])
                else:
                    distance_calculator = Euclidean(NeuralEmbedding(TrueSummariesComputation()))
        else:
            raise NotImplementedError

        # perform ABC inference (if needed) and compute performance metrics:
        if perform_ABC:
            if ABC_eps is None:
                # this determines the starting epsilon value considering the distances between pairs of generated
                # data; # wouldn't it be better considering the actual observation? Would that lead to bias?
                # I don't think so.
                eps = determine_eps(samples_matrix_test[-100:], distance_calculator, epsilon_quantile)
            else:  # in this case we use the provided numerical epsilon value, discarding the epsilon_quantile
                eps = ABC_eps
            start = time()
            jrnl = ABC_inference(ABC_algorithm, ABC_model, x_obs, distance_calculator, eps, n_samples,
                                 ABC_steps, backend, seed=seed, full_output=1 if ABC_full_output else 0,
                                 **kwargs_abc_inference)
            running_time_inference_ABC[obs_index] = time() - start
            np.save(inference_folder + "running_time_inference" + namefile_postfix_no_index,
                    running_time_inference_ABC)
            if save_jrnl:
                jrnl.save(inference_folder + "jrnl" + namefile_postfix + ".jnl")

        params, weights = extract_params_and_weights_from_journal(jrnl)
        means = extract_posterior_mean_from_journal(jrnl)
        params_subsample = params
        params_subsample, weights_subsample = subsample_trace_and_weights(params, weights, size=subsample_size)

        if true_posterior_available:
            # wass dist and RMSE:
            wass_dist_ABC[obs_index] = wass_dist(params_subsample, trace_true_subsample,
                                                 weights_post_1=weights_subsample)
            RMSE_post_mean_ABC[obs_index] = np.linalg.norm(means - true_post_means)

            np.save(inference_folder + "wass_dist" + namefile_postfix_no_index, wass_dist_ABC)
            np.save(inference_folder + "RMSE_post_mean" + namefile_postfix_no_index, RMSE_post_mean_ABC)

    else:
        raise NotImplementedError

    # produce plots:
    if plot_marginal_densities:
        plot_single_marginal_with_trace_samples(theta_obs, params, trace_true if true_posterior_available else None,
                                                weights_trace_approx=weights if inference_technique == "ABC" else None,
                                                namefile=inference_folder + "single_marginals" + namefile_postfix + ".png",
                                                param_names=param_names,
                                                thetarange=np.array([lower_bounds, upper_bounds]))
    if plot_bivariate_densities:
        fig, ax = plot_bivariate_marginal_with_trace_samples(
            theta_obs, params, trace_true_subsample if true_posterior_available else None,
            weights_trace_approx=weights if inference_technique == "ABC" else None,
            thetarange=np.array([lower_bounds, upper_bounds]),
            color="#40739E", figsize_vertical=5, title_size=16, label_size=16, param1_name=param_names[0],
            param2_name=param_names[1], vertical=True, space_subplots=0.23)
        bbox_inches = Bbox(np.array([[0, .4], [5, 9.2]]))
        plt.savefig(inference_folder + "bivariate_marginals" + namefile_postfix + ".png", bbox_inches=bbox_inches)
        plt.close()
    if plot_journal and inference_technique == "ABC":
        jrnl.plot_posterior_distr(path_to_save=inference_folder + "joint_posterior" + namefile_postfix + ".png",
                                  true_parameter_values=theta_obs)
    if plot_trace_flag and inference_technique == "exchange":
        if generate_new_MCMC:
            plot_trace(trace_exchange, theta_dim, param_names,
                       namefile=inference_folder + "trace" + namefile_postfix + ".png", burnin=burnin_exchange_MCMC)
        else:
            plot_trace(trace_exchange_burned_in, theta_dim, param_names,
                       namefile=inference_folder + "trace" + namefile_postfix + ".png")

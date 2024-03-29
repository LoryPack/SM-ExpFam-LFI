import os
import pickle
import sys
from time import sleep, time

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.getcwd())

from abcpy.NN_utilities.utilities import save_net
from abcpy.backends import BackendDummy, BackendMPI
from abcpy.continuousmodels import Uniform
from abcpy.statisticslearning import ExponentialFamilyScoreMatching as ExpFamStatistics
from abcpy.statistics import Identity
from abcpy.inferences import DrawFromPrior

from src.utils_Lorenz95_example import StochLorenz95

from src.training_routines import FP_training_routine
from src.functions import scale_samples, plot_losses, save_dict_to_json, DummyScaler
from src.networks import createDefaultNN, createDefaultNNWithDerivatives, create_PEN_architecture
from src.utils_arma_example import ARMAmodel
from src.utils_gaussian_example import generate_gaussian_training_samples
from src.utils_gamma_example import generate_gamma_training_samples
from src.utils_beta_example import generate_beta_training_samples
from src.parsers import train_net_batch_parser

parser = train_net_batch_parser(default_root_folder=None)

args = parser.parse_args()
technique = args.technique
model = args.model
save_train_data = args.save_train_data
load_train_data = args.load_train_data
sleep_time = args.sleep
epochs = args.epochs
no_scheduler = args.no_scheduler
results_folder = args.root_folder
nets_folder = args.nets_folder
datasets_folder = args.datasets_folder
noise_sliced = args.noise_sliced
var_red_sliced = not args.no_var_red_sliced
batch_norm_last_layer = not args.no_bn
affine_batch_norm = args.affine_bn
SM_lr = args.lr_data
FP_lr = args.lr_data if args.lr_data is not None else 0.001  # we also use this as a learning rate for the FP approach
SM_lr_theta = args.lr_theta
batch_size = args.batch_size
early_stopping = not args.no_early_stop
update_batchnorm_running_means_before_eval = args.update_batchnorm_running_means_before_eval
momentum = args.bn_momentum
epochs_before_early_stopping = args.epochs_before_early_stopping
epochs_test_interval = args.epochs_test_interval
use_MPI = args.use_MPI
generate_data_only = args.generate_data_only
save_net_at_each_epoch = args.save_net_at_each_epoch

# checks
if model not in (
        "gaussian", "beta", "gamma", "MA2", "AR2", "fullLorenz95",
        "fullLorenz95smaller") or technique not in (
        "SM", "SSM", "FP"):
    raise NotImplementedError

backend = BackendMPI() if use_MPI else BackendDummy()

if generate_data_only:
    print("Generate data only, no train.")
else:
    print("{} model with {}.".format(model, technique))
# set up the default root folder and other values
default_root_folder = {"gaussian": "results/gaussian/",
                       "gamma": "results/gamma/",
                       "beta": "results/beta/",
                       "AR2": "results/AR2/",
                       "MA2": "results/MA2/",
                       "fullLorenz95": "results/fullLorenz95/",
                       "fullLorenz95smaller": "results/fullLorenz95smaller/"}
if results_folder is None:
    results_folder = default_root_folder[model]
if nets_folder is None:
    nets_folder = "net-SM" if technique == "SM" else "net-FP"

results_folder = results_folder + '/'
nets_folder = results_folder + nets_folder + '/'
datasets_folder = results_folder + datasets_folder + '/'

if SM_lr is None:
    SM_lr = 0.001
if SM_lr_theta is None:
    SM_lr_theta = 0.001

if sleep_time > 0:
    print("Wait for {} minutes...".format(sleep_time))
    sleep(60 * sleep_time)
    print("Done waiting!")

seed = 42

cuda = True
torch.set_num_threads(4)

n_samples_training = 10 ** 4
n_samples_evaluation = 10 ** 4

save_net_flag = True
lam = 0

args_dict = args.__dict__
# add other arguments to the config dicts
args_dict['seed'] = seed
args_dict['n_samples_training'] = n_samples_training
args_dict['n_samples_evaluation'] = n_samples_evaluation
args_dict['lr_FP_actual'] = FP_lr
args_dict['lr_data_actual'] = SM_lr
args_dict['lr_theta_actual'] = SM_lr_theta
args_dict['batch_size'] = batch_size
args_dict['save_net'] = save_net_flag
args_dict['cuda'] = cuda

save_dict_to_json(args.__dict__, nets_folder + 'config.json')

if model == "gaussian":

    mu_bounds = [-10, 10]
    sigma_bounds = [1, 10]

    args_dict['mu_bounds'] = mu_bounds
    args_dict['sigma_bounds'] = sigma_bounds
    start = time()
    # generate training data
    theta_vect, samples_matrix = generate_gaussian_training_samples(n_theta=n_samples_training,
                                                                    size_iid_samples=10, seed=seed,
                                                                    mu_bounds=mu_bounds,
                                                                    sigma_bounds=sigma_bounds)
    print("Data generation took {:.4f} seconds".format(time() - start))

    scaler_data_FP = MinMaxScaler().fit(
        samples_matrix.reshape(-1, samples_matrix.shape[
            -1]))
    lower_bound = upper_bound = None
    scale_samples_flag = True
    scale_parameters_flag = True

    # generate test data for using early stopping in learning the statistics with SM
    theta_vect_test, samples_matrix_test = generate_gaussian_training_samples(n_theta=n_samples_evaluation,
                                                                              size_iid_samples=10,
                                                                              mu_bounds=mu_bounds,
                                                                              sigma_bounds=sigma_bounds)

elif model == "beta":

    alpha_bounds = [1, 3]
    beta_bounds = [1, 3]

    args_dict['alpha_bounds'] = alpha_bounds
    args_dict['beta_bounds'] = beta_bounds
    start = time()
    theta_vect, samples_matrix = generate_beta_training_samples(n_theta=n_samples_training,
                                                                size_iid_samples=10, seed=seed,
                                                                alpha_bounds=alpha_bounds,
                                                                beta_bounds=beta_bounds)
    print("Data generation took {:.4f} seconds".format(time() - start))

    scaler_data_FP = MinMaxScaler().fit(
        samples_matrix.reshape(-1, samples_matrix.shape[-1]))  # fit the transformation scaler!
    lower_bound = np.array([0] * 10)
    upper_bound = np.array([1] * 10)
    scale_samples_flag = True
    scale_parameters_flag = True

    # generate test data for using early stopping in learning the statistics with SM
    theta_vect_test, samples_matrix_test = generate_beta_training_samples(n_theta=n_samples_evaluation,
                                                                          size_iid_samples=10,
                                                                          alpha_bounds=alpha_bounds,
                                                                          beta_bounds=beta_bounds)

elif model == "gamma":

    k_bounds = [1, 3]
    theta_bounds = [1, 3]

    args_dict['k_bounds'] = k_bounds
    args_dict['theta_bounds'] = theta_bounds
    start = time()
    theta_vect, samples_matrix = generate_gamma_training_samples(n_theta=n_samples_training,
                                                                 size_iid_samples=10, seed=seed,
                                                                 k_bounds=k_bounds,
                                                                 theta_bounds=theta_bounds)
    print("Data generation took {:.4f} seconds".format(time() - start))
    scaler_data_FP = MinMaxScaler().fit(
        samples_matrix.reshape(-1, samples_matrix.shape[-1]))  # fit the transformation scaler!
    lower_bound = np.array([0] * 10)
    upper_bound = None
    scale_samples_flag = True
    scale_parameters_flag = True

    # generate test data for using early stopping in learning the statistics with SM
    theta_vect_test, samples_matrix_test = generate_gamma_training_samples(n_theta=n_samples_evaluation,
                                                                           size_iid_samples=10,
                                                                           k_bounds=k_bounds,
                                                                           theta_bounds=theta_bounds)

elif model == "AR2":
    arma_size = 100
    ar1_bounds = [-1, 1]
    ar2_bounds = [-1, 0]
    args_dict['arma_size'] = arma_size
    args_dict['ar1_bounds'] = ar1_bounds
    args_dict['ar2_bounds'] = ar2_bounds

    ar1 = Uniform([[ar1_bounds[0]], [ar1_bounds[1]]], name='ar1')
    ar2 = Uniform([[ar2_bounds[0]], [ar2_bounds[1]]], name='ar2')
    arma_abc_model = ARMAmodel([ar1, ar2], num_AR_params=2, num_MA_params=0, size=arma_size)

    if not load_train_data:
        print("Generating data... ({} samples in total)".format(n_samples_training + n_samples_evaluation))
        start = time()
        draw_from_prior = DrawFromPrior([arma_abc_model], backend=backend, seed=seed)
        theta_vect, samples_matrix = draw_from_prior.sample_par_sim_pairs(n_samples_training, 1)
        theta_vect_test, samples_matrix_test = draw_from_prior.sample_par_sim_pairs(n_samples_evaluation, 1)
        samples_matrix = samples_matrix.reshape(samples_matrix.shape[0], samples_matrix.shape[-1])
        samples_matrix_test = samples_matrix_test.reshape(samples_matrix_test.shape[0], samples_matrix_test.shape[-1])
        print("Data generation took {:.2f} seconds".format(time() - start))
        if save_train_data:
            # save data before scalers are applied.
            np.save(datasets_folder + "theta_vect.npy", theta_vect)
            np.save(datasets_folder + "samples_matrix.npy", samples_matrix)
            np.save(datasets_folder + "theta_vect_test.npy", theta_vect_test)
            np.save(datasets_folder + "samples_matrix_test.npy", samples_matrix_test)
    else:
        theta_vect = np.load(datasets_folder + "theta_vect.npy", allow_pickle=True)
        samples_matrix = np.load(datasets_folder + "samples_matrix.npy", allow_pickle=True)
        theta_vect_test = np.load(datasets_folder + "theta_vect_test.npy", allow_pickle=True)
        samples_matrix_test = np.load(datasets_folder + "samples_matrix_test.npy", allow_pickle=True)
        print("Loaded data; {} training samples, {} test samples".format(theta_vect.shape[0], theta_vect_test.shape[0]))

    # no scalers here
    scaler_data_FP = DummyScaler()
    lower_bound = upper_bound = None
    scale_samples_flag = False
    scale_parameters_flag = False

elif model == "MA2":
    arma_size = 100
    ma1_bounds = [-1, 1]
    ma2_bounds = [0, 1]
    args_dict['arma_size'] = arma_size
    args_dict['ma1_bounds'] = ma1_bounds
    args_dict['ma2_bounds'] = ma2_bounds

    ma1 = Uniform([[ma1_bounds[0]], [ma1_bounds[1]]], name='ma1')
    ma2 = Uniform([[ma2_bounds[0]], [ma2_bounds[1]]], name='ma2')
    arma_abc_model = ARMAmodel([ma1, ma2], num_AR_params=0, num_MA_params=2, size=arma_size)

    if not load_train_data:
        print("Generating data... ({} samples in total)".format(n_samples_training + n_samples_evaluation))
        start = time()
        draw_from_prior = DrawFromPrior([arma_abc_model], backend=backend, seed=seed)
        theta_vect, samples_matrix = draw_from_prior.sample_par_sim_pairs(n_samples_training, 1)
        theta_vect_test, samples_matrix_test = draw_from_prior.sample_par_sim_pairs(n_samples_evaluation, 1)
        samples_matrix = samples_matrix.reshape(samples_matrix.shape[0], samples_matrix.shape[-1])
        samples_matrix_test = samples_matrix_test.reshape(samples_matrix_test.shape[0], samples_matrix_test.shape[-1])
        print("Data generation took {:.2f} seconds".format(time() - start))
        if save_train_data:
            # save data before scalers are applied.
            np.save(datasets_folder + "theta_vect.npy", theta_vect)
            np.save(datasets_folder + "samples_matrix.npy", samples_matrix)
            np.save(datasets_folder + "theta_vect_test.npy", theta_vect_test)
            np.save(datasets_folder + "samples_matrix_test.npy", samples_matrix_test)
    else:
        theta_vect = np.load(datasets_folder + "theta_vect.npy", allow_pickle=True)
        samples_matrix = np.load(datasets_folder + "samples_matrix.npy", allow_pickle=True)
        theta_vect_test = np.load(datasets_folder + "theta_vect_test.npy", allow_pickle=True)
        samples_matrix_test = np.load(datasets_folder + "samples_matrix_test.npy", allow_pickle=True)
        print("Loaded data; {} training samples, {} test samples".format(theta_vect.shape[0], theta_vect_test.shape[0]))

    # no scalers here
    scaler_data_FP = DummyScaler()
    lower_bound = upper_bound = None
    scale_samples_flag = False
    scale_parameters_flag = False

if "Lorenz95" in model:
    # here use a larger set of summaries (23). Same parameter range for theta1, theta2 as for Hakk stats experiment,
    # but here we also add sigma and phi parameters.

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

    if not load_train_data:
        lorenz = StochLorenz95([theta1, theta2, sigma_e, phi], time_units=1.5 if model == "fullLorenz95smaller" else 4,
                               n_timestep_per_time_unit=30, K=8 if model == "fullLorenz95smaller" else 40,
                               name='lorenz')

        print("Generating data... ({} samples in total)".format(n_samples_training + n_samples_evaluation))
        start = time()
        # give seed here; we do not put any scaler for the timeseries data
        draw_from_prior = DrawFromPrior([lorenz], backend=backend, seed=seed)
        theta_vect, samples_matrix = draw_from_prior.sample_par_sim_pairs(n_samples_training, 1)
        samples_matrix = samples_matrix.reshape(samples_matrix.shape[0], samples_matrix.shape[-1])
        print(samples_matrix.shape, sys.getsizeof(samples_matrix))
        # Size of the tensor: 3 MB for the train set with 200 observations.Then 20000 -> 300 MB. Need to save that.
        theta_vect_test, samples_matrix_test = draw_from_prior.sample_par_sim_pairs(n_samples_evaluation, 1)
        samples_matrix_test = samples_matrix_test.reshape(samples_matrix_test.shape[0], samples_matrix_test.shape[-1])

        print("Data generation took {:.2f} seconds".format(time() - start))

        if save_train_data:
            # save data before scalers are applied.
            np.save(datasets_folder + "theta_vect.npy", theta_vect)
            np.save(datasets_folder + "samples_matrix.npy", samples_matrix)
            np.save(datasets_folder + "theta_vect_test.npy", theta_vect_test)
            np.save(datasets_folder + "samples_matrix_test.npy", samples_matrix_test)

    else:
        theta_vect = np.load(datasets_folder + "theta_vect.npy", allow_pickle=True)
        theta_vect_test = np.load(datasets_folder + "theta_vect_test.npy", allow_pickle=True)
        print("Loading data")
        samples_matrix = np.load(datasets_folder + "samples_matrix.npy", allow_pickle=True)
        samples_matrix_test = np.load(datasets_folder + "samples_matrix_test.npy", allow_pickle=True)
        print(samples_matrix.shape)
        print("Loaded data; {} training samples, {} test samples".format(theta_vect.shape[0], theta_vect_test.shape[0]))

    if model == "fullLorenz95":
        lower_bound = np.array([None] * 4800)
        upper_bound = np.array([None] * 4800)
    else:
        lower_bound = np.array([None] * 360)
        upper_bound = np.array([None] * 360)
    scaler_data_FP = DummyScaler()
    scale_samples_flag = False
    scale_parameters_flag = True

# update the n samples with the actual ones (if we loaded them from saved datasets).
args_dict['n_samples_training'] = theta_vect.shape[0]
args_dict['n_samples_evaluation'] = theta_vect_test.shape[0]
args_dict['scale_samples'] = str(type(scale_samples_flag))
args_dict['scale_parameters'] = str(type(scale_parameters_flag))

save_dict_to_json(args.__dict__, nets_folder + 'config.json')

if generate_data_only:
    print("Generating data has finished")
    exit()

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
                                                affine_batch_norm=affine_batch_norm,
                                                batch_norm_last_layer_momentum=momentum)
    net_FP_architecture = createDefaultNN(10, 2, [30, 50, 50, 20], nonlinearity=nonlinearity())
    # net_FP_architecture = createDefaultNN(10, 2, [30, 50, 50, 20], nonlinearity=torch.nn.ReLU())
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

    net_theta_SM_architecture = createDefaultNN(2, 2, [15, 30, 30, 15], nonlinearity=nonlinearity(),
                                                batch_norm_last_layer=batch_norm_last_layer,
                                                affine_batch_norm=affine_batch_norm,
                                                batch_norm_last_layer_momentum=momentum)

elif model == "AR2":
    nonlinearity = torch.nn.Softplus
    phi_net_data_architecture = createDefaultNNWithDerivatives(3, 20, [50, 50, 30], nonlinearity=nonlinearity)
    rho_net_data_architecture = createDefaultNNWithDerivatives(22, 3, [50, 50], nonlinearity=nonlinearity)
    net_data_SM_architecture = create_PEN_architecture(phi_net_data_architecture, rho_net_data_architecture, 2)

    phi_net_FP_architecture = createDefaultNN(3, 20, [50, 50, 30], nonlinearity=nonlinearity())
    rho_net_FP_architecture = createDefaultNN(22, 2, [50, 50], nonlinearity=nonlinearity())
    net_FP_architecture = create_PEN_architecture(phi_net_FP_architecture, rho_net_FP_architecture, 2)

    net_theta_SM_architecture = createDefaultNN(2, 2, [15, 30, 30, 15], nonlinearity=nonlinearity(),
                                                batch_norm_last_layer=batch_norm_last_layer,
                                                affine_batch_norm=affine_batch_norm,
                                                batch_norm_last_layer_momentum=momentum)

elif "fullLorenz95" in model:
    nonlinearity = torch.nn.Softplus
    if model == "fullLorenz95":
        phi_net_data_SM_architecture = createDefaultNNWithDerivatives(80, 20, [120, 160, 120],
                                                                      nonlinearity=nonlinearity)
        rho_net_data_SM_architecture = createDefaultNNWithDerivatives(60, 5, [80, 100, 80], nonlinearity=nonlinearity)
        net_data_SM_architecture = create_PEN_architecture(phi_net_data_SM_architecture, rho_net_data_SM_architecture,
                                                           order=1, number_timeseries=40)
        phi_net_FP_architecture = createDefaultNNWithDerivatives(80, 20, [120, 160, 120], nonlinearity=nonlinearity)
        rho_net_FP_architecture = createDefaultNNWithDerivatives(60, 4, [80, 100, 80], nonlinearity=nonlinearity)
        net_FP_architecture = create_PEN_architecture(phi_net_FP_architecture, rho_net_FP_architecture,
                                                      order=1, number_timeseries=40)
    else:
        phi_net_data_SM_architecture = createDefaultNNWithDerivatives(16, 20, [50, 100, 50], nonlinearity=nonlinearity)
        rho_net_data_SM_architecture = createDefaultNNWithDerivatives(28, 5, [40, 90, 35], nonlinearity=nonlinearity)
        net_data_SM_architecture = create_PEN_architecture(phi_net_data_SM_architecture, rho_net_data_SM_architecture,
                                                           order=1, number_timeseries=8)
        phi_net_data_FP_architecture = createDefaultNNWithDerivatives(16, 20, [50, 100, 50], nonlinearity=nonlinearity)
        rho_net_data_FP_architecture = createDefaultNNWithDerivatives(28, 4, [40, 90, 35], nonlinearity=nonlinearity)
        net_FP_architecture = create_PEN_architecture(phi_net_data_FP_architecture, rho_net_data_FP_architecture,
                                                      order=1, number_timeseries=8)

    net_theta_SM_architecture = createDefaultNN(4, 4, [30, 50, 50, 30], nonlinearity=nonlinearity(),
                                                batch_norm_last_layer=batch_norm_last_layer,
                                                affine_batch_norm=affine_batch_norm,
                                                batch_norm_last_layer_momentum=momentum)

# TRAIN THE NETS:
start = time()
if technique in ["SM", "SSM"]:
    if seed is not None:
        torch.manual_seed(seed)
    # define networks
    net_data_SM = net_data_SM_architecture()
    net_theta_SM = net_theta_SM_architecture()

    # convert the simulations and parameters to numpy if they are torch objects:
    theta_vect = theta_vect.numpy() if isinstance(theta_vect, torch.Tensor) else theta_vect
    theta_vect_test = theta_vect_test.numpy() if isinstance(theta_vect_test, torch.Tensor) else theta_vect_test
    samples_matrix = samples_matrix.numpy() if isinstance(samples_matrix, torch.Tensor) else samples_matrix
    samples_matrix_test = samples_matrix_test.numpy() if isinstance(samples_matrix_test,
                                                                    torch.Tensor) else samples_matrix_test

    statistics_learning = ExpFamStatistics(
        model=None, statistics_calc=Identity(), backend=BackendDummy(),  # backend and model are not used 
        simulations_net=net_data_SM, parameters_net=net_theta_SM, parameters=theta_vect,
        simulations=samples_matrix, parameters_val=theta_vect_test, simulations_val=samples_matrix_test,
        scale_samples=scale_samples_flag, scale_parameters=scale_parameters_flag,
        lower_bound_simulations=lower_bound, upper_bound_simulations=upper_bound,
        sliced=technique == "SSM", noise_type=noise_sliced,
        variance_reduction=var_red_sliced and not noise_sliced == "sphere",
        n_epochs=epochs, batch_size=batch_size, lr_simulations=SM_lr,
        lr_parameters=SM_lr_theta, seed=seed, start_epoch_early_stopping=epochs_before_early_stopping,
        epochs_early_stopping_interval=epochs_test_interval, early_stopping=early_stopping,
        scheduler_parameters=False if no_scheduler else None,
        scheduler_simulations=False if no_scheduler else None,
        cuda=cuda, lam=lam, batch_norm_update_before_test=update_batchnorm_running_means_before_eval)

    loss_list = statistics_learning.train_losses
    test_loss_list = statistics_learning.test_losses

    scaler_data = statistics_learning.get_simulations_scaler()
    scaler_theta = statistics_learning.get_parameters_scaler()

    if save_net_flag:
        save_net(nets_folder + "net_theta_SM.pth", net_theta_SM)
        save_net(nets_folder + "net_data_SM.pth", net_data_SM)
        pickle.dump(scaler_data, open(nets_folder + "scaler_data_SM.pkl", "wb"))
        pickle.dump(scaler_theta, open(nets_folder + "scaler_theta_SM.pkl", "wb"))
    args_dict['scaler_data'] = str(type(scaler_data))
    args_dict['scaler_theta'] = str(type(scaler_theta))

if technique == "FP":
    if seed is not None:
        torch.manual_seed(seed)
    net_FP = net_FP_architecture()
    samples_matrix_rescaled = scale_samples(scaler_data_FP, samples_matrix)
    samples_matrix_test_rescaled = scale_samples(scaler_data_FP, samples_matrix_test, requires_grad=True)

    if isinstance(theta_vect, np.ndarray):
        theta_vect = torch.tensor(theta_vect, dtype=torch.float)
    if isinstance(theta_vect_test, np.ndarray):
        theta_vect_test = torch.tensor(theta_vect_test, dtype=torch.float)
    loss_list, test_loss_list = FP_training_routine(
        samples_matrix_rescaled, theta_vect, net_FP, samples_matrix_test=samples_matrix_test_rescaled,
        theta_vect_test=theta_vect_test, n_epochs=epochs, batch_size=batch_size, lr=FP_lr, seed=seed,
        return_loss_list=True, return_test_loss_list=True, epochs_test_interval=epochs_test_interval,
        early_stopping=early_stopping, enable_scheduler=not no_scheduler, cuda=cuda,
        epochs_before_early_stopping=epochs_before_early_stopping)

    if save_net_flag:
        save_net(nets_folder + "net_data_FP.pth", net_FP)
        pickle.dump(scaler_data_FP, open(nets_folder + "scaler_data_FP.pkl", "wb"))
training_time = time() - start

plot_losses(loss_list, test_loss_list, nets_folder + "losses.png")
np.save(nets_folder + "loss.npy", np.array(loss_list))
np.save(nets_folder + "test_loss.npy", np.array(test_loss_list))

text_file = open(nets_folder + "training_time.txt", "w")
string = "Training time: {:.2f} seconds.".format(training_time)
text_file.write(string + "\n")
text_file.close()

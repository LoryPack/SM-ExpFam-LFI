import os
import pickle
import sys
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.transforms import Bbox
from sklearn.decomposition import PCA

sys.path.append(os.getcwd())

from abcpy.NN_utilities.utilities import load_net
from abcpy.continuousmodels import Uniform

from src.utils_Lorenz95_example import StochLorenz95

import seaborn as sns

from src.networks import createDefaultNN
from src.functions import scale_thetas, generate_training_samples_ABC_model, DummyScaler
from src.parsers import parser_plot_stats
from src.utils_gaussian_example import generate_gaussian_training_samples, TrueNaturalParametersComputationGaussian
from src.utils_gamma_example import generate_gamma_training_samples, TrueNaturalParametersComputationGamma
from src.utils_beta_example import generate_beta_training_samples, TrueNaturalParametersComputationBeta
from src.mcc import compute_mcc
from src.utils_arma_example import ARMAmodel

os.environ["QT_QPA_PLATFORM"] = 'offscreen'

parser = parser_plot_stats(None, default_nets_folder="net")
args = parser.parse_args()

model = args.model
sleep_time = args.sleep
n_observations = args.n_observations
results_folder = args.root_folder
nets_folder = args.nets_folder
batch_norm_last_layer = not args.no_bn
affine_batch_norm = args.affine_bn
seed = args.seed

if model not in ("gaussian", "beta", "gamma", "MA2", "AR2", "Lorenz95", "fullLorenz95"):
    raise NotImplementedError

print("{} model.".format(model))
default_root_folder = {"gaussian": "results/gaussian/",
                       "gamma": "results/gamma/",
                       "beta": "results/beta/",
                       "AR2": "results/AR2/",
                       "MA2": "results/MA2/",
                       "Lorenz95": "results/Lorenz95/",
                       "fullLorenz95": "results/fullLorenz95/"}
if results_folder is None:
    results_folder = default_root_folder[model]

results_folder = results_folder + '/'
nets_folder = results_folder + nets_folder + '/'

if sleep_time > 0:
    print("Wait for {} minutes...".format(sleep_time))
    sleep(60 * sleep_time)
    print("Done waiting!")

R_training = 1

if model == "gaussian":
    mu_bounds = [-10, 10]
    sigma_bounds = [1, 10]

    theta_vect, _ = generate_gaussian_training_samples(n_theta=n_observations,
                                                       size_iid_samples=10, seed=seed, mu_bounds=mu_bounds,
                                                       sigma_bounds=sigma_bounds)

    true_nat_params_test = TrueNaturalParametersComputationGaussian()(theta_vect).numpy()

elif model == "gamma":
    k_bounds = [1, 3]
    theta_bounds = [1, 3]

    theta_vect, _ = generate_gamma_training_samples(n_theta=n_observations,
                                                    size_iid_samples=10, seed=seed, k_bounds=k_bounds,
                                                    theta_bounds=theta_bounds)
    true_nat_params_test = TrueNaturalParametersComputationGamma()(theta_vect).numpy()

elif model == "beta":
    alpha_bounds = [1, 3]
    beta_bounds = [1, 3]

    theta_vect, _ = generate_beta_training_samples(n_theta=n_observations,
                                                   size_iid_samples=10, seed=seed, alpha_bounds=alpha_bounds,
                                                   beta_bounds=beta_bounds)
    true_nat_params_test = TrueNaturalParametersComputationBeta()(theta_vect).numpy()

elif model == "MA2":
    arma_size = 100
    ma1_bounds = [-1, 1]
    ma2_bounds = [0, 1]

    ma1 = Uniform([[ma1_bounds[0]], [ma1_bounds[1]]], name='ma1')
    ma2 = Uniform([[ma2_bounds[0]], [ma2_bounds[1]]], name='ma2')
    arma_abc_model = ARMAmodel([ma1, ma2], num_AR_params=0, num_MA_params=2, size=arma_size)

    theta_vect, _ = generate_training_samples_ABC_model(arma_abc_model, n_observations, seed=seed)

elif model == "AR2":
    arma_size = 100
    ar1_bounds = [-1, 1]
    ar2_bounds = [-1, 0]

    ar1 = Uniform([[ar1_bounds[0]], [ar1_bounds[1]]], name='ar1')
    ar2 = Uniform([[ar2_bounds[0]], [ar2_bounds[1]]], name='ar2')
    arma_abc_model = ARMAmodel([ar1, ar2], num_AR_params=2, num_MA_params=0, size=arma_size)
    theta_vect, _ = generate_training_samples_ABC_model(arma_abc_model, n_observations, seed=seed)

elif model in ["Lorenz95", "fullLorenz95", ]:
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
    lorenz = StochLorenz95([theta1, theta2, sigma_e, phi], time_units=1, n_timestep_per_time_unit=1, name='lorenz')
    theta_vect, _ = generate_training_samples_ABC_model(lorenz, n_observations, seed=seed)

scaler_theta_SM = pickle.load(open(nets_folder + "scaler_theta_SM.pkl", "rb"))
if scaler_theta_SM is None:
    scaler_theta_SM = DummyScaler()
theta_vect_rescaled = scale_thetas(scaler_theta_SM, theta_vect)

if model in ("gaussian", "gamma", "beta"):
    nonlinearity = torch.nn.Softplus
    output_size = 2
    # net_theta_SM_architecture = createDefaultNN(2, output_size, [30, 50, 50, 20], nonlinearity=nonlinearity(),
    net_theta_SM_architecture = createDefaultNN(2, output_size, [15, 30, 30, 15], nonlinearity=nonlinearity(),
                                                batch_norm_last_layer=batch_norm_last_layer,
                                                affine_batch_norm=affine_batch_norm)

elif model in ("MA2", "AR2"):
    nonlinearity = torch.nn.Softplus
    output_size = 2
    net_theta_SM_architecture = createDefaultNN(2, output_size, [15, 30, 30, 15], nonlinearity=nonlinearity(),
                                                # net_theta_SM_architecture = createDefaultNN(2, output_size, [15, 30, 30, 15], nonlinearity=nonlinearity(),
                                                batch_norm_last_layer=batch_norm_last_layer,
                                                affine_batch_norm=affine_batch_norm)
elif model == "Lorenz95":
    # define network architectures:
    nonlinearity = torch.nn.Softplus
    output_size = 4
    net_theta_SM_architecture = createDefaultNN(4, output_size, [30, 50, 50, 30], nonlinearity=nonlinearity(),
                                                batch_norm_last_layer=batch_norm_last_layer,
                                                affine_batch_norm=affine_batch_norm)

elif model == "fullLorenz95":
    nonlinearity = torch.nn.Softplus
    output_size = 4
    net_theta_SM_architecture = createDefaultNN(4, 4, [30, 50, 50, 30], nonlinearity=nonlinearity(),
                                                batch_norm_last_layer=batch_norm_last_layer,
                                                affine_batch_norm=affine_batch_norm)

net_theta_SM = load_net(nets_folder + "net_theta_SM.pth", net_theta_SM_architecture).eval()

learned_params_test = net_theta_SM(
    theta_vect_rescaled.reshape(-1, theta_vect_rescaled.shape[-1])).detach().numpy()

# compute now a measure of correlation between the learned embeddings:
corr_matrix = np.corrcoef(learned_params_test, rowvar=False)
mean_embedding_correlation = 2 * np.sum(np.abs(np.tril(corr_matrix, k=-1))) / (output_size * (output_size - 1))
print("Mean correlation between embeddings: {:4f}".format(mean_embedding_correlation))
np.save(nets_folder + "mean_embedding_correlation_params", mean_embedding_correlation)

# perform now PCA on the learned embeddings and try to get the number of components; we can either:
# 1) keep the whole number of components and then produce scree plot to consider the explained variance
# 2) use n_components='mle' for automatically finding the n components. Not so useful here maybe.
pca = PCA().fit(learned_params_test)
# produce scree plot:
plt.bar(np.arange(learned_params_test.shape[1], ), pca.explained_variance_ratio_)
plt.xlabel("Component")
plt.ylabel("Variance explained")
plt.savefig(nets_folder + "scree_plot_parameters.png")
plt.close()

if learned_params_test.shape[1] > 2:
    # use a pairplot for the statistics:
    sns.pairplot(pd.DataFrame(learned_params_test))
    plt.savefig(nets_folder + "learned_parameters.png")
else:
    plt.plot(learned_params_test[:, 0], learned_params_test[:, 1], ".")
    plt.xlabel("First learned parameters")
    plt.ylabel("Second learned parameters")
    plt.savefig(nets_folder + "learned_parameters.png")
plt.close()
if model in ("beta", "gamma", "gaussian"):
    # compute the MCC:
    mcc_strong_in, mcc_strong_out, mcc_weak_in, mcc_weak_out = compute_mcc(true_nat_params_test,
                                                                           learned_params_test[:, 0:2], weak=True)
    np.save(nets_folder + "mcc_strong_in_params", mcc_strong_in)
    np.save(nets_folder + "mcc_strong_out_params", mcc_strong_out)
    np.save(nets_folder + "mcc_weak_in_params", mcc_weak_in)
    np.save(nets_folder + "mcc_weak_out_params", mcc_weak_out)

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 8), sharex="col", sharey="row")
    ax[0, 0].plot(true_nat_params_test[:, 0], learned_params_test[:, 0], ".")
    # ax[0, 0].set_xlabel("First true parameters")
    ax[0, 0].set_ylabel("First learned parameters")
    ax[0, 1].plot(true_nat_params_test[:, 1], learned_params_test[:, 0], ".")
    # ax[0, 1].set_xlabel("Second true parameters")
    # ax[0, 1].set_ylabel("First learned parameters")
    ax[1, 0].plot(true_nat_params_test[:, 0], learned_params_test[:, 1], ".")
    ax[1, 0].set_xlabel("First true parameters")
    ax[1, 0].set_ylabel("Second learned parameters")
    ax[1, 1].plot(true_nat_params_test[:, 1], learned_params_test[:, 1], ".")
    ax[1, 1].set_xlabel("Second true parameters")
    # ax[1, 1].set_ylabel("Second learned parameters")
    # ax[0,0].ticklabel_format(axis='both', scilimits=(-2,2))
    # ax[0,1].ticklabel_format(axis='both', scilimits=(-2,2))
    # ax[1,0].ticklabel_format(axis='both', scilimits=(-2,2))
    # ax[1,1].ticklabel_format(axis='both', scilimits=(-2,2))
    plt.subplots_adjust(wspace=0, hspace=0)
    bbox_inches = Bbox(np.array([[-0.1, .35], [7.25, 7.1]]))
    plt.savefig(nets_folder + "learned_true_parameters.pdf", bbox_inches=bbox_inches)
    plt.close()
    # fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
    # ax[0, 0].loglog(true_nat_params_test[:, 0], learned_params_test[:, 0], ".")
    # ax[0, 0].set_xlabel("First true parameters")
    # ax[0, 0].set_ylabel("First learned parameters")
    # ax[0, 1].loglog(true_nat_params_test[:, 1], learned_params_test[:, 0], ".")
    # ax[0, 1].set_xlabel("Second true parameters")
    # ax[0, 1].set_ylabel("First learned parameters")
    # ax[1, 0].loglog(true_nat_params_test[:, 0], learned_params_test[:, 1], ".")
    # ax[1, 0].set_xlabel("First true parameters")
    # ax[1, 0].set_ylabel("Second learned parameters")
    # ax[1, 1].loglog(true_nat_params_test[:, 1], learned_params_test[:, 1], ".")
    # ax[1, 1].set_xlabel("Second true parameters")
    # ax[1, 1].set_ylabel("Second learned parameters")
    # plt.savefig(nets_folder + "learned_true_parameters_loglog.png")
    # plt.close()

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

from src.networks import createDefaultNN, create_PEN_architecture
from src.functions import scale_samples, generate_training_samples_ABC_model
from src.parsers import parser_plot_stats
from src.utils_beta_example import generate_beta_training_samples, TrueSummariesComputationBeta
from src.utils_gamma_example import generate_gamma_training_samples, TrueSummariesComputationGamma
from src.utils_gaussian_example import generate_gaussian_training_samples, TrueSummariesComputationGaussian
from src.utils_Lorenz95_example import LorenzLargerStatistics
from src.mcc import compute_mcc
from src.utils_arma_example import ARMAmodel

os.environ["QT_QPA_PLATFORM"] = 'offscreen'

parser = parser_plot_stats(None, default_nets_folder="net-SM")
parser.add_argument('--FP', action="store_true", )
args = parser.parse_args()

model = args.model
sleep_time = args.sleep
n_observations = args.n_observations
results_folder = args.root_folder
nets_folder = args.nets_folder
batch_norm_last_layer = not args.no_bn
affine_batch_norm = args.affine_bn
FP = args.FP
seed = args.seed

if model not in ("gaussian", "beta", "gamma", "MA2", "AR2", "Lorenz95"):
    raise NotImplementedError

print("{} model.".format(model))
default_root_folder = {"gaussian": "results/gaussian/",
                       "gamma": "results/gamma/",
                       "beta": "results/beta/",
                       "AR2": "results/AR2/",
                       "MA2": "results/MA2/",
                       "Lorenz95": "results/Lorenz95/"}
if results_folder is None:
    results_folder = default_root_folder[model]

results_folder = results_folder + '/'
nets_folder = results_folder + nets_folder + '/'

if sleep_time > 0:
    print("Wait for {} minutes...".format(sleep_time))
    sleep(60 * sleep_time)
    print("Done waiting!")

# seed = 12345
R_training = 1

if model == "beta":
    alpha_bounds = [1, 3]
    beta_bounds = [1, 3]

    theta_vect, samples_matrix = generate_beta_training_samples(n_theta=n_observations, 
                                                                size_iid_samples=10, seed=seed,
                                                                alpha_bounds=alpha_bounds,
                                                                beta_bounds=beta_bounds)

    true_suff_stat_test = TrueSummariesComputationBeta()(samples_matrix.reshape(-1, 10)).numpy()

elif model == "gamma":
    k_bounds = [1, 3]
    theta_bounds = [1, 3]

    theta_vect, samples_matrix = generate_gamma_training_samples(n_theta=n_observations, 
                                                                 size_iid_samples=10, seed=seed, k_bounds=k_bounds,
                                                                 theta_bounds=theta_bounds)
    true_suff_stat_test = TrueSummariesComputationGamma()(samples_matrix.reshape(-1, 10)).numpy()

elif model == "gaussian":
    mu_bounds = [-10, 10]
    sigma_bounds = [1, 10]

    theta_vect, samples_matrix = generate_gaussian_training_samples(n_theta=n_observations, 
                                                                    size_iid_samples=10, seed=seed, mu_bounds=mu_bounds,
                                                                    sigma_bounds=sigma_bounds)
    true_suff_stat_test = TrueSummariesComputationGaussian()(samples_matrix.reshape(-1, 10)).numpy()

elif model == "MA2":
    arma_size = 100
    ma1_bounds = [-1, 1]
    ma2_bounds = [0, 1]

    ma1 = Uniform([[ma1_bounds[0]], [ma1_bounds[1]]], name='ma1')
    ma2 = Uniform([[ma2_bounds[0]], [ma2_bounds[1]]], name='ma2')
    arma_abc_model = ARMAmodel([ma1, ma2], num_AR_params=0, num_MA_params=2, size=arma_size)

    theta_vect, samples_matrix = generate_training_samples_ABC_model(arma_abc_model, n_observations, seed=seed)

elif model == "AR2":
    arma_size = 100
    ar1_bounds = [-1, 1]
    ar2_bounds = [-1, 0]

    ar1 = Uniform([[ar1_bounds[0]], [ar1_bounds[1]]], name='ar1')
    ar2 = Uniform([[ar2_bounds[0]], [ar2_bounds[1]]], name='ar2')
    arma_abc_model = ARMAmodel([ar1, ar2], num_AR_params=2, num_MA_params=0, size=arma_size)
    theta_vect, samples_matrix = generate_training_samples_ABC_model(arma_abc_model, n_observations, seed=seed)

elif model == "Lorenz95":
    # we follow here the same setup as in the ratio estimation paper. Then we only infer theta_1, theta_2 and keep
    # fixed sigma_3 and phi. We moreover change prior range and use the Hakkarainen statistics
    statistics = LorenzLargerStatistics(degree=1, cross=False)  # add cross-terms

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

    lorenz = StochLorenz95([theta1, theta2, sigma_e, phi], time_units=4, n_timestep_per_time_unit=30, name='lorenz')
    theta_vect, samples_matrix = generate_training_samples_ABC_model(lorenz, n_observations, seed=seed)

    samples_matrix = statistics.statistics([sample for sample in samples_matrix])  # 21-dimensional stat
    print(samples_matrix.shape)

if model in ("gaussian", "beta", "gamma"):
    # define network architectures:
    nonlinearity = torch.nn.Softplus
    # nonlinearity = torch.nn.Tanhshrink
    output_size = 3 if not FP else 2
    net_data_SM_architecture = createDefaultNN(10, output_size, [30, 50, 50, 20], nonlinearity=nonlinearity(),
                                               batch_norm_last_layer=batch_norm_last_layer,
                                               affine_batch_norm=affine_batch_norm)
    net_data_FP_architecture = net_data_SM_architecture
elif model == "MA2":
    # define network architectures:
    nonlinearity = torch.nn.Softplus
    output_size = 3 if not FP else 2
    phi_net_data_SM_architecture = createDefaultNN(11, 20, [50, 50, 30], nonlinearity=nonlinearity())
    rho_net_data_SM_architecture = createDefaultNN(30, output_size, [50, 50], nonlinearity=nonlinearity(),
                                                   batch_norm_last_layer=batch_norm_last_layer,
                                                   affine_batch_norm=affine_batch_norm)
    net_data_SM_architecture = create_PEN_architecture(phi_net_data_SM_architecture, rho_net_data_SM_architecture, 10)
    net_data_FP_architecture = net_data_SM_architecture

elif model == "AR2":
    # define network architectures:
    nonlinearity = torch.nn.Softplus
    output_size = 3 if not FP else 2
    phi_net_data_SM_architecture = createDefaultNN(3, 20, [50, 50, 30], nonlinearity=nonlinearity())
    rho_net_data_SM_architecture = createDefaultNN(22, output_size, [50, 50], nonlinearity=nonlinearity(),
                                                   batch_norm_last_layer=batch_norm_last_layer,
                                                   affine_batch_norm=affine_batch_norm)
    net_data_SM_architecture = create_PEN_architecture(phi_net_data_SM_architecture, rho_net_data_SM_architecture, 2)
    net_data_FP_architecture = net_data_SM_architecture

elif model == "Lorenz95":
    nonlinearity = torch.nn.Softplus
    output_size = 5 if not FP else 4
    net_data_SM_architecture = createDefaultNN(23, output_size, [70, 120, 120, 70, 20], nonlinearity=nonlinearity(),
                                               batch_norm_last_layer=batch_norm_last_layer,
                                               affine_batch_norm=affine_batch_norm)
    net_data_FP_architecture = net_data_SM_architecture

if FP:
    scaler_data_FP = pickle.load(open(nets_folder + "scaler_data_FP.pkl", "rb"))
    net_data_FP = load_net(nets_folder + "net_data_FP.pth", net_data_FP_architecture).eval()
    samples_matrix_rescaled = scale_samples(scaler_data_FP, samples_matrix)
    learned_stats_test = net_data_FP(
        samples_matrix_rescaled.reshape(-1, samples_matrix_rescaled.shape[-1])).detach().numpy()
else:
    scaler_data_SM = pickle.load(open(nets_folder + "scaler_data_SM.pkl", "rb"))
    net_data_SM = load_net(nets_folder + "net_data_SM.pth", net_data_SM_architecture).eval()
    samples_matrix_rescaled = scale_samples(scaler_data_SM, samples_matrix)
    learned_stats_test = net_data_SM(
        samples_matrix_rescaled.reshape(-1, samples_matrix_rescaled.shape[-1])).detach().numpy()

# compute now a measure of correlation between the learned embeddings:
corr_matrix = np.corrcoef(learned_stats_test, rowvar=False)
mean_embedding_correlation = 2 * np.sum(np.abs(np.tril(corr_matrix, k=-1))) / (output_size * (output_size - 1))
print("Mean correlation between embeddings: {:4f}".format(mean_embedding_correlation))
np.save(nets_folder + "mean_embedding_correlation", mean_embedding_correlation)

# perform now PCA on the learned embeddings and try to get the number of components; we can either:
# 1) keep the whole number of components and then produce scree plot to consider the explained variance
# 2) use n_components='mle' for automatically finding the n components. Not so useful here maybe.
pca = PCA().fit(learned_stats_test)
# produce scree plot:
plt.bar(np.arange(learned_stats_test.shape[1], ), pca.explained_variance_ratio_)
plt.xlabel("Component")
plt.ylabel("Variance explained")
plt.savefig(nets_folder + "scree_plot_statistics.png")
plt.close()

if learned_stats_test.shape[1] > 2:
    # use a pairplot for the statistics:
    sns.pairplot(pd.DataFrame(learned_stats_test))
    plt.savefig(nets_folder + "learned_statistics.png")
else:
    plt.plot(learned_stats_test[:, 0], learned_stats_test[:, 1], ".")
    plt.xlabel("First learned statistics")
    plt.ylabel("Second learned statistics")
    # plt.savefig(nets_folder + "learned_statistics_{}.png".format(i))
    plt.savefig(nets_folder + "learned_statistics.png")
plt.close()
if model in ("beta", "gamma", "gaussian"):  # the models for which you know the true embedding:
    # compute mcc
    mcc_strong_in, mcc_strong_out, mcc_weak_in, mcc_weak_out = compute_mcc(true_suff_stat_test,
                                                                           learned_stats_test[:, 0:2], weak=True)
    np.save(nets_folder + "mcc_strong_in", mcc_strong_in)
    np.save(nets_folder + "mcc_strong_out", mcc_strong_out)
    np.save(nets_folder + "mcc_weak_in", mcc_weak_in)
    np.save(nets_folder + "mcc_weak_out", mcc_weak_out)

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 8), sharex="col", sharey="row")
    ax[0, 0].plot(true_suff_stat_test[:, 0], learned_stats_test[:, 0], ".")
    # ax[0, 0].set_xlabel("First true statistics")
    ax[0, 0].set_ylabel("First learned statistics")
    ax[0, 1].plot(true_suff_stat_test[:, 1], learned_stats_test[:, 0], ".")
    # ax[0, 1].set_xlabel("Second true statistics")
    # ax[0, 1].set_ylabel("First learned statistics")
    ax[1, 0].plot(true_suff_stat_test[:, 0], learned_stats_test[:, 1], ".")
    ax[1, 0].set_xlabel("First true statistics")
    ax[1, 0].set_ylabel("Second learned statistics")
    ax[1, 1].plot(true_suff_stat_test[:, 1], learned_stats_test[:, 1], ".")
    ax[1, 1].set_xlabel("Second true statistics")
    # ax[1, 1].set_ylabel("Second learned statistics")
    # ax[0, 0].ticklabel_format(axis='both', scilimits=(-2, 2))
    # ax[0, 1].ticklabel_format(axis='both', scilimits=(-2, 2))
    # ax[1, 0].ticklabel_format(axis='both', scilimits=(-2, 2))
    # ax[1, 1].ticklabel_format(axis='both', scilimits=(-2, 2))
    plt.subplots_adjust(wspace=0, hspace=0)
    bbox_inches = Bbox(np.array([[-0.1, .35], [7.25, 7.1]]))
    plt.savefig(nets_folder + "learned_true_statistics.pdf", bbox_inches=bbox_inches)
    # plt.savefig(nets_folder + "learned_true_statistics_{}.png".format(i))
    plt.close()

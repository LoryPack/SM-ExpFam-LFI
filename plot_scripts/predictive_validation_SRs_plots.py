import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

sys.path.append(os.getcwd())

from src.functions import plot_confidence_bands_performance_vs_iteration

parser = argparse.ArgumentParser()
parser.add_argument('model',
                    help="The statistical model to consider.")
parser.add_argument('--sleep', type=float, default=0, help='Minutes to sleep before starting')
parser.add_argument('--root_folder', type=str, default=None)
parser.add_argument('--inference_folder_ABC_FP', type=str, default=None, help="If not provided, I will "
                                                                              "not use this in the plot")
parser.add_argument('--inference_folder_ABC_SM', type=str, default=None, help="If not provided, I will "
                                                                              "not use this in the plot")
parser.add_argument('--inference_folder_ABC_SSM', type=str, default=None, help="If not provided, I will "
                                                                               "not use this in the plot")
parser.add_argument('--inference_folder_Exc_SM', type=str, default=None, help="If not provided, I will "
                                                                              "not use this in the plot")
parser.add_argument('--inference_folder_Exc_SSM', type=str, default=None, help="If not provided, I will "
                                                                               "not use this in the plot")
parser.add_argument('--ABC_FP_algorithm', type=str, default="SABC")
parser.add_argument('--ABC_FP_n_samples', type=int, default=1000)
parser.add_argument('--ABC_FP_steps', type=int, default=100)
parser.add_argument('--ABC_SM_algorithm', type=str, default="SABC")
parser.add_argument('--ABC_SM_steps', type=int, default=100)
parser.add_argument('--ABC_SM_n_samples', type=int, default=1000)
parser.add_argument('--ABC_SSM_algorithm', type=str, default="SABC")
parser.add_argument('--ABC_SSM_steps', type=int, default=100)
parser.add_argument('--ABC_SSM_n_samples', type=int, default=1000)
parser.add_argument('--Exc_SM_n_samples', type=int, default=10000)
parser.add_argument('--Exc_SSM_n_samples', type=int, default=10000)
parser.add_argument('--CI_level', type=float, default=95,
                    help="The size of confidence interval (CI) to produce the plots. It represents the confidence "
                         "intervals in the plots vs iterations, and the position of the whiskers in the boxplots.")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--subsample_size', type=int, default=1000,
                    help='Number of samples used for bivariate plots and Wass distance(if required).')

args = parser.parse_args()
model = args.model
sleep_time = args.sleep
results_folder = args.root_folder
seed = args.seed
subsample_size = args.subsample_size
ABC_FP_algorithm = args.ABC_FP_algorithm
ABC_FP_steps = args.ABC_FP_steps
ABC_FP_n_samples = args.ABC_FP_n_samples
ABC_SM_algorithm = args.ABC_SM_algorithm
ABC_SM_steps = args.ABC_SM_steps
ABC_SM_n_samples = args.ABC_SM_n_samples
ABC_SSM_algorithm = args.ABC_SSM_algorithm
ABC_SSM_steps = args.ABC_SSM_steps
ABC_SSM_n_samples = args.ABC_SSM_n_samples
Exc_SM_n_samples = args.Exc_SM_n_samples
Exc_SSM_n_samples = args.Exc_SSM_n_samples
CI_level = args.CI_level

np.random.seed(seed)  # seed rng

# checks
if model not in ("fullLorenz95", "fullLorenz95smaller"):
    raise NotImplementedError
print(f"{model} model.")

default_root_folder = {"fullLorenz95": "results/fullLorenz95/",
                       "fullLorenz95smaller": "results/fullLorenz95smaller/"}
if results_folder is None:
    results_folder = default_root_folder[model]

model_text = {"fullLorenz95": "Large Lorenz96", "fullLorenz95smaller": "Small Lorenz96"}

# now load all SR values; we load only SR results where the folder for the corresponding technique has beeen provided

ABC_FP_namefile_postfix = f"_{ABC_FP_algorithm}_FP_steps_{ABC_FP_steps}_n_samples_{ABC_FP_n_samples}"
ABC_SM_namefile_postfix = f"_{ABC_SM_algorithm}_SM_steps_{ABC_SM_steps}_n_samples_{ABC_SM_n_samples}"
ABC_SSM_namefile_postfix = f"_{ABC_SSM_algorithm}_SSM_steps_{ABC_SSM_steps}_n_samples_{ABC_SSM_n_samples}"
Exc_SM_namefile_postfix = "_{}_n_samples_{}".format("exchange", Exc_SM_n_samples)
Exc_SSM_namefile_postfix = "_{}_n_samples_{}".format("exchange", Exc_SSM_n_samples)

all_techniques = ["ABC-FP", "ABC-SM", "ABC-SSM", "Exc-SM", "Exc-SSM"]
all_folders = [args.inference_folder_ABC_FP, args.inference_folder_ABC_SM, args.inference_folder_ABC_SSM,
               args.inference_folder_Exc_SM, args.inference_folder_Exc_SSM]
all_namefile_postfixs = [ABC_FP_namefile_postfix, ABC_SM_namefile_postfix, ABC_SSM_namefile_postfix,
                         Exc_SM_namefile_postfix, Exc_SSM_namefile_postfix]
techniques = []
energy_timestep_list = []
energy_cumulative_list = []
kernel_timestep_list = []
kernel_cumulative_list = []

for n_tech, tech in enumerate(all_techniques):
    folder = all_folders[n_tech]
    if folder is not None:
        folder = results_folder + "/" + folder + "/"
        namefile_postfix = all_namefile_postfixs[n_tech]
        techniques.append(all_techniques[n_tech])
        energy_timestep_list.append(np.load(folder + "energy_sr_values_timestep" + namefile_postfix + ".npy"))
        energy_cumulative_list.append(np.load(folder + "energy_sr_values_cumulative" + namefile_postfix + ".npy"))
        kernel_timestep_list.append(np.load(folder + "kernel_sr_values_timestep" + namefile_postfix + ".npy"))
        kernel_cumulative_list.append(np.load(folder + "kernel_sr_values_cumulative" + namefile_postfix + ".npy"))

list_indeces = (np.arange(len(techniques)) + 1).tolist()
labelsize = 24
ticksize_x = 16
ticksize_y = 12

# plot 1: boxplots for cumulative SR values:
fig, ax1 = plt.subplots(1, figsize=(0.8 * len(list_indeces) + 1, 6))
ax2 = ax1.twinx()
ax1.set_title(f"{model_text[model]}", size=labelsize)
c = "C1"
ax1.boxplot(energy_cumulative_list, whis=(50 - CI_level / 2, 50 + CI_level / 2), patch_artist=True, notch=True,
            boxprops=dict(facecolor=c, color=c, alpha=0.5),
            capprops=dict(color=c), whiskerprops=dict(color=c), flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=c), widths=0.3, positions=np.array(list_indeces) - 0.15)
ax1.set_xticks(list_indeces)
ax1.tick_params(axis='y', labelcolor=c)
ax1.set_xticklabels(techniques, rotation=45)
ax1.set_ylabel("Energy Score", color=c, size=labelsize)
# ax1.set_ylim(ymin=0)
c = "C2"
ax2.boxplot(kernel_cumulative_list, whis=(50 - CI_level / 2, 50 + CI_level / 2), patch_artist=True, notch=True,
            boxprops=dict(facecolor=c, color=c, alpha=0.5),
            capprops=dict(color=c), whiskerprops=dict(color=c), flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=c), widths=0.3, positions=np.array(list_indeces) + 0.15)
ax2.set_ylabel("Kernel Score", color=c, size=labelsize)
ax2.tick_params(axis='y', labelcolor=c)
ax2.set_xticks(list_indeces)
ax2.set_xticklabels(techniques, rotation=45)
# ax2.set_ylim(ymin=0)
ax1.tick_params(axis='y', which='both', labelsize=ticksize_y, )
ax2.tick_params(axis='y', which='both', labelsize=ticksize_y, )
ax1.tick_params(axis='x', which='both', labelsize=ticksize_x, )
ax2.tick_params(axis='x', which='both', labelsize=ticksize_x, )
bbox_inches = Bbox(np.array([[-1, -0.5], [0.8 * len(list_indeces) + 2, 5.7]]))
plt.savefig(results_folder + "SR_cumulative_boxplot.pdf", bbox_inches=bbox_inches)
plt.close()

# plot 2 (timeseries of SRs)
bbox_inches = Bbox(np.array([[-0.1, -0.15], [5.5, 3.8]]))
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
fig2, ax2 = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))

for n_tech, tech in enumerate(techniques):
    energy_timestep = energy_timestep_list[n_tech]
    kernel_timestep = kernel_timestep_list[n_tech]
    c = f"C{n_tech}"
    plot_confidence_bands_performance_vs_iteration(energy_timestep.transpose(1, 0), fig=fig, ax=ax, label=tech,
                                                   color_band_1=c, color_line=c, band_1=CI_level, alpha_1=0.3,
                                                   alpha_2=0, alpha_3=0, fill_between=True)
    plot_confidence_bands_performance_vs_iteration(kernel_timestep.transpose(1, 0), fig=fig2, ax=ax2, label=tech,
                                                   color_band_1=c, color_line=c, band_1=CI_level, alpha_1=0.3,
                                                   alpha_2=0, alpha_3=0, fill_between=True)

for a in [ax, ax2]:
    a.set_title(f"{model_text[model]}")
    a.set_xlabel(r"$t$")
    if model == "fullLorenz95smaller":
        a.set_xticks([0, 14, 29, 44])
        a.set_xticklabels([0, 0.5, 1, 1.5])
        a.set_xlim(xmin=0, xmax=44)
    else:
        a.set_xticks([0, 29, 59, 89, 119])
        a.set_xticklabels([0, 1, 2, 3, 4])
        a.set_xlim(xmin=0, xmax=119)
    a.legend()
ax.set_ylabel("Energy Score")
ax2.set_ylabel("Kernel Score")

fig.savefig(results_folder + "SR_timeseries_energy.pdf", bbox_inches=bbox_inches)
fig2.savefig(results_folder + "SR_timeseries_kernel.pdf", bbox_inches=bbox_inches)
plt.close()

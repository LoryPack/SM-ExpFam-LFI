#!/bin/bash

# We put here all commands needed to obtain the results presented in the paper. We remark that running this bash file
# as a single script may not be a good idea, as the different python scripts in here may have very different
# computational demands (some may require parallel computing, and you may choose to use GPU to train the NNs)
# Moreover, all Python scripts below have additional options, which you can list by:
#       python3 <file.py> -h

# CREATE TREE STRUCTURE:
mkdir results
mkdir results/fullLorenz95
mkdir results/fullLorenz95/ABC-FP
mkdir results/fullLorenz95/ABC-SSM
mkdir results/fullLorenz95/net-FP
mkdir results/fullLorenz95/net-SSM
mkdir results/fullLorenz95/observations

# GENERATE OBSERVATIONS AND EXACT POSTERIORS

n_observations=100
python3 scripts/generate_obs.py fullLorenz95 --n_observations $n_observations

# TRAIN THE NNs
python3 scripts/train_net.py SSM fullLorenz95 --nets_folder net-SSM --epochs 1000 --lr_data 0.001 --lr_theta 0.001 \
    --update --bn_mom 0.9  --epochs_before_early_stopping 500 --epochs_test_interval 50 --load_train_data --dataset datasets_10000
python3 scripts/train_net.py FP fullLorenz95 --nets_folder net-FP --epochs 1000 --lr_data 0.001  --epochs_before_early_stopping 200 --epochs_test_interval 25 --load_train_data --dataset datasets_10000

# PRODUCE PLOTS FOR NN EMBEDDINGS
python3 plot_scripts/plot_learned_stats.py fullLorenz95 --nets_folder net-SSM --no_bn --n_obs 100
python3 plot_scripts/plot_learned_nat_params.py fullLorenz95 --nets_folder net-SSM --n_obs 100

python3 plot_scripts/plot_learned_stats.py fullLorenz95 --nets_folder net-FP --no_bn --n_obs 100 --FP

# INFERENCES WITH ABC-SSM and ABC-FP; this uses MPI to parallelize, with number of tasks given by NTASKS
NTASKS=8  # adapt this to your setup
start_obs=0
ABC_algorithm=SABC
ABC_steps=100
n_samples=1000
ABC_eps=100000000
n_observations=100
SABC_cutoff=0  # increase this for faster stop.

technique=FP 
inference_folder=ABC-FP
nets_folder=net-FP

mpirun -n $NTASKS python3 scripts/inferences.py $technique $model \
         --use_MPI \
         --inference_technique ABC \
         --start_observation_index $start_obs \
         --n_observations $n_observations \
         --ABC_alg $ABC_algorithm \
         --ABC_steps $ABC_steps \
         --n_samples $n_samples \
         --inference_folder $inference_folder \
         --nets_folder $nets_folder \
         --ABC_full_output \
         --ABC_eps $ABC_eps \
         --SABC_cutoff $SABC_cutoff \
         --load_trace_if_available \
         --no_weighted_eucl_dist \
         --seed 42

technique=SSM
inference_folder=ABC-SSM
nets_folder=net-SSM

mpirun -n $NTASKS python3 scripts/inferences.py $technique $model \
         --use_MPI \
         --inference_technique ABC \
         --start_observation_index $start_obs \
         --n_observations $n_observations \
         --ABC_alg $ABC_algorithm \
         --ABC_steps $ABC_steps \
         --n_samples $n_samples \
         --inference_folder $inference_folder \
         --nets_folder $nets_folder \
         --ABC_full_output \
         --ABC_eps $ABC_eps \
         --SABC_cutoff $SABC_cutoff \
         --load_trace_if_available \
         --seed 42 

# COMPUTE THE PREDICTIVE PERFORMANCE WITH SCORING RULES:
mpirun -n 8 python3 scripts/predictive_validation_SRs.py SSM ${model} --inference_technique ABC --inference_folder ABC-SSM --ABC_steps 100 --use_MPI --gamma_kernel_score 6.384898984255503
mpirun -n 8 python3 scripts/predictive_validation_SRs.py FP ${model} --inference_technique ABC --inference_folder ABC-FP --ABC_steps 100 --use_MPI --gamma_kernel_score 6.384898984255503

# final scoring rules plot:
python3 plot_scripts/predictive_validation_SRs_plots.py ${model} --inference_folder_ABC_SSM ABC-SSM --inference_folder_ABC_FP ABC-FP

# PLOTS
python3 plot_scripts/plot_Lorenz_multiv_post.py fullLorenz95 FP --inference_folder ABC-FP --inference_technique ABC
python3 plot_scripts/plot_Lorenz_multiv_post.py fullLorenz95 SSM --inference_folder ABC-SSM --inference_technique ABC


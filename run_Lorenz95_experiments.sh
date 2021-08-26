#!/bin/bash

# We put here all commands needed to obtain the results presented in the paper. We remark that running this bash file
# as a single script may not be a good idea, as the different python scripts in here may have very different
# computational demands (some may require parallel computing, and you may choose to use GPU to train the NNs)
# Moreover, all Python scripts below have additional options, which you can list by:
#       python3 <file.py> -h

# CREATE TREE STRUCTURE:
mkdir results
mkdir results/Lorenz95
mkdir results/Lorenz95/ABC-FP
mkdir results/Lorenz95/ABC-SM
mkdir results/Lorenz95/Exc-SM
mkdir results/Lorenz95/net_FP
mkdir results/Lorenz95/net_SM
mkdir results/Lorenz95/observations

# GENERATE OBSERVATIONS AND EXACT POSTERIORS

n_observations=100
python3 scripts/generate_obs.py Lorenz95 --n_observations $n_observations

# TRAIN THE NNs
python3 scripts/train_net.py SM Lorenz95 --nets_folder net-SM --epochs 1000 --lr_data 0.001 --lr_theta 0.001 \
    --update --bn_mom 0.9  --epochs_before_early_stopping 500 --epochs_test_interval 50
python3 scripts/train_net.py FP Lorenz95 --nets_folder net-FP --epochs 1000 --lr_data 0.001  --epochs_before_early_stopping 200 --epochs_test_interval 25

# PRODUCE PLOTS FOR NN EMBEDDINGS
python3 plot_scripts/plot_learned_stats.py Lorenz95 --nets_folder net-SM --no_bn --n_obs 1000
python3 plot_scripts/plot_learned_nat_params.py Lorenz95 --nets_folder net-SM --n_obs 1000

python3 plot_scripts/plot_learned_stats.py Lorenz95 --nets_folder net-FP --no_bn --n_obs 1000 --FP

# INFERENCES WITH Exc-SM

model=Lorenz95
prop_size=0.1
K=200  # briding steps
inner_steps=400
inf_f=Exc-SM
burnin=10000
n_samples=50000
net_f=net-SM
tune_window=100 #000000

python3 scripts/inferences.py SM ${model} --burnin $burnin --n_samples $n_samples \
    --inference_folder $inf_f --nets_f $net_f \
    --start 0 --n_obs $n_observations \
    --aux_MCMC_inner_steps_exchange_MCMC $inner_steps --bridging ${K} \
    --aux_MCMC_proposal_size_exchange_MCMC ${prop_size} \
    --tuning ${tune_window} \
    --deb warn \
    --propose_new_theta_exchange_MCMC transformation


# INFERENCES WITH ABC-SM and ABC-FP; this uses MPI to parallelize, with number of tasks given by NTASKS
NTASKS=4  # adapt this to your setup
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

technique=SM 
inference_folder=ABC-SM
nets_folder=net-SM

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

# PRODUCE PLOTS
python3 plot_scripts/timeseries_predictive_error.py Lorenz95 exc --exchange_folder Exc-SM --n_observations 100

python3 likelihood_experiments/plots/plot_Lorenz_multiv_post.py FP --inference_folder ABC-FP --inference_technique ABC
python3 likelihood_experiments/plots/plot_Lorenz_multiv_post.py SM --inference_folder ABC-SM --inference_technique ABC
python3 likelihood_experiments/plots/plot_Lorenz_multiv_post.py SM --inference_folder Exc-SM --inference_technique exchange


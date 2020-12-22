#!/bin/bash

# We put here all commands needed to obtain the results presented in the paper. We remark that running this bash file
# as a single script may not be a good idea, as the different python scripts in here may have very different
# computational demands (some may require parallel computing, and you may choose to use GPU to train the NNs)
# Moreover, all Python scripts below have additional options, which you can list by:
#       python3 <file.py> -h

# CREATE TREE STRUCTURE: 
mkdir results
mkdir results/MA2
mkdir results/MA2/ABC-FP
mkdir results/MA2/ABC-SM
mkdir results/MA2/Exc-SM
mkdir results/MA2/Exc-SM/10_inn_steps
mkdir results/MA2/Exc-SM/30_inn_steps
mkdir results/MA2/Exc-SM/100_inn_steps
mkdir results/MA2/Exc-SM/200_inn_steps
mkdir results/MA2/net_FP
mkdir results/MA2/net_SM
mkdir results/MA2/observations
mkdir results/MA2/PMC-RE
mkdir results/MA2/PMC-SL

# GENERATE OBSERVATIONS AND EXACT POSTERIORS

n_observations=100
python3 scripts/generate_obs.py MA2 --n_observations $n_observations

# TRAIN THE NNs
python3 scripts/train_net.py SM MA2 --nets_folder net-SM --epochs 500 --lr_data 0.001 --lr_theta 0.001 \
    --update --bn_mom 0.9 --epochs_before_early_stopping 100 --epochs_test_interval 25
python3 scripts/train_net.py FP MA2 --nets_folder net-FP --epochs 1000 --lr_data 0.001 --epochs_before_early_stopping 500 --epochs_test_interval 25

# PRODUCE PLOTS FOR NN EMBEDDINGS
python3 plot_scripts/plot_learned_stats.py MA2 --nets_folder net-SM --no_bn --n_obs 1000
python3 plot_scripts/plot_learned_nat_params.py MA2 --nets_folder net-SM --n_obs 1000

python3 plot_scripts/plot_learned_stats.py MA2 --nets_folder net-FP --no_bn --n_obs 1000 --FP

# INFERENCES WITH Exc-SM WITH DIFFERENT NUMBER OF INNER MCMC STEPS

model=MA2
inn_steps_values=( 10 30 100 200 )

for ((k=0;k<${#inn_steps_values[@]};++k)); do

    prop_size=0.1
    K=0  # briding steps
    inner_steps=${inn_steps_values[k]}
    inf_f=Exc-SM/${inner_steps}_inn_steps
    burnin=10000
    n_samples=10000
    net_f=net-SM
    tune_window=100 #000000

    python3 scripts/inferences.py SM ${model} --burnin $burnin --n_samples $n_samples \
        --inference_folder $inf_f --nets_f $net_f \
        --start 0 --n_obs $n_observations \
        --aux_MCMC_inner_steps_exchange_MCMC $inner_steps --bridging ${K} \
        --aux_MCMC_proposal_size_exchange_MCMC ${prop_size} \
        --tuning ${tune_window} \
        --deb warn \
        --propose_new_theta_exchange_MCMC truncnorm

done

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

# INFERENCES WITH PMC-RE, PMR-SL

start_obs=0
steps=10
n_samples=1000
n_observations=100

n_samples_per_param=100
technique=SL
inference_folder=PMC-SL
mpirun -n $NTASKS python3 scripts/SL_RE_experiments.py $technique $model \
         --use_MPI \
         --start_obs $start_obs \
         --n_observations $n_observations \
         --steps $steps \
         --n_samples $n_samples \
         --n_samples_per_param $n_samples_per_param \
         --inference_folder $inference_folder \
         --full_output \
         --load_trace_if_available \
         --seed 2


n_samples_per_param=1000
technique=RE
inference_folder=PMC-RE
mpirun -n $NTASKS python3 scripts/SL_RE_experiments.py $technique $model \
         --use_MPI \
         --start_obs $start_obs \
         --n_observations $n_observations \
         --steps $steps \
         --n_samples $n_samples \
         --n_samples_per_param $n_samples_per_param \
         --inference_folder $inference_folder \
         --full_output \
         --load_trace_if_available \
         --seed 2


# PRODUCE PLOTS
# compare the different number inner MCMC steps:
python3 plot_scripts/wass_dist_diff_inner_MCMC_steps.py MA2 --n_samples 10000
# single bivariate plot for the observation in the paper:
python3 plot_scripts/plot_bivariate.py MA2 --inference_folder Exc-SM/30_inn_steps --start 3 --n_obs 4

# final wass distance plot:
n_observations=100
python3 plot_scripts/compare_inference_results_diff_techniques.py MA2 final --n_obs $n_observations --subsample_size 10000 --inference_folder_exchange_SM Exc-SM/30_inn_steps --load_exchange_SM_if_available

# Wass dist plots vs iterations; these may take a while
python3 plot_scripts/compare_inference_results_diff_techniques.py MA2 ABC --n_obs $n_observations --subsample_size 1000 --inference_folder_exchange_SM Exc-SM/30_inn_steps --load_exchange_SM_if_available
python3 plot_scripts/compare_inference_results_diff_techniques.py MA2 SL --n_obs $n_observations --subsample_size 1000 --inference_folder_exchange_SM Exc-SM/30_inn_steps --load_exchange_SM_if_available
python3 plot_scripts/compare_inference_results_diff_techniques.py MA2 RE --n_obs $n_observations --subsample_size 1000 --inference_folder_exchange_SM Exc-SM/30_inn_steps --load_exchange_SM_if_available

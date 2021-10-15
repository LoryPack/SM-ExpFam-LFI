import argparse


def parser_generate_obs():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        help="The statistical model to consider; can be 'AR2', 'MA2', 'beta', 'gamma', 'gaussian', 'fullLorenz95' or 'fullLorenz95smaller'")
    parser.add_argument('--root_folder', type=str, default=None)
    parser.add_argument('--observation_folder', type=str, default="observations")
    parser.add_argument('--start_observation', type=int, default=0, help='Observation from which to start the script')
    parser.add_argument('--n_observations', type=int, default=10, help='Total number of observations')
    parser.add_argument('--sleep', type=float, default=0, help='Minutes to sleep before starting')

    return parser


def train_net_batch_parser(default_root_folder, default_nets_folder=None, default_lr_data=None, default_lr_theta=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('technique', type=str, help="The technique to use; can be 'SM' or 'SSM' (both using the "
                                                    "exponential family but with two different losses) or 'FP'.")
    parser.add_argument('model', type=str, help="The statistical model to consider.")
    parser.add_argument('--sleep', type=float, default=0, help='Minutes to sleep before starting (default 0).')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs (defatul 500).')
    parser.add_argument('--no_scheduler', action="store_true", help="Disable scheduler")
    parser.add_argument('--root_folder', type=str, default=default_root_folder)
    parser.add_argument('--nets_folder', type=str, default=default_nets_folder)
    parser.add_argument('--noise_sliced', type=str, default="radermacher",
                        help="Which kind of noise to use with sliced SM; can be radermacher, sphere or gaussian. "
                             "Default is radermacher.")
    parser.add_argument('--no_var_red_sliced', action="store_true",
                        help="Do not use the variance reduction approach with sliced SM or not."
                             " Notice that the variance reduction is never used whene noise_sliced is 'sphere'")
    parser.add_argument('--no_bn', action="store_true")
    parser.add_argument('--affine_bn', action="store_true")
    parser.add_argument('--lr_data', type=float, default=default_lr_data, help='Learning rate for data')
    parser.add_argument('--lr_theta', type=float, default=default_lr_theta, help='Learning rate for theta')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size (default 1000).')
    parser.add_argument('--no_early_stop', action="store_true")
    parser.add_argument('--update_batchnorm_running_means_before_eval', action="store_true")
    parser.add_argument('--bn_momentum', type=float, default=0.1, )
    parser.add_argument('--epochs_before_early_stopping', type=int, default=200)
    parser.add_argument('--epochs_test_interval', type=int, default=10)
    parser.add_argument('--save_train_data', action="store_true",
                        help="Whether to store or not the generated datasets. This only works for Lorenz example.")
    parser.add_argument('--load_train_data', action="store_true",
                        help="Whether to load the generated datasets. This only works for Lorenz example.")
    parser.add_argument('--datasets_folder', type=str, default="observations",
                        help="Where to store training dataset. For Lorenz example")
    parser.add_argument('--use_MPI', '-m', action="store_true",
                        help="Use MPI distribution to generate the observation. Notice that this slows down a lot NN "
                             "training. Therefore, it is suggested to be used together with --generate_data_only")
    parser.add_argument('--generate_data_only', action="store_true",
                        help="It stops after generating data without starting training.")
    parser.add_argument('--save_net_at_each_epoch', action="store_true")

    return parser


def parser_approx_likelihood_approach(default_root_folder, default_nets_folder="net-SM",
                                      default_observation_folder="observations",
                                      default_inference_folder="Exc-SM", default_n_observations=10,
                                      default_n_samples=1000, default_burnin_MCMC=10000):
    """Returns a parser with some arguments. It is still possible to add further arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('technique', type=str,
                        help="The technique to use; can be 'SM' or 'SSM' (both using exponential family),"
                        "or 'FP'. The latter does not work with exchange.")
    parser.add_argument('model', type=str, help="The statistical model to consider.")
    parser.add_argument('--inference_technique', type=str, default="exchange",
                        help="Inference approach; can be exchange or ABC.")
    parser.add_argument('--sleep', type=float, default=0, help='Minutes to sleep before starting')
    parser.add_argument('--start_observation_index', type=int, default=0, help='Index to start from')
    parser.add_argument('--n_observations', type=int, default=default_n_observations,
                        help='Total number of observations.')
    parser.add_argument('--n_samples', type=int, default=default_n_samples,
                        help='Number of samples (for ABC or exchange MCMC)')
    parser.add_argument('--burnin_MCMC', type=int, default=default_burnin_MCMC,
                        help='Burnin samples for exchange MCMC.')
    parser.add_argument('--aux_MCMC_inner_steps_exchange_MCMC', type=int, default=0,
                        help='Number of MCMC steps in the inner chain in exchange MCMC')
    parser.add_argument('--aux_MCMC_proposal_size_exchange_MCMC', type=float, default=0.1,
                        help='Proposal size for auxiliary MCMC chain inside ExchangeMCMC')
    parser.add_argument('--load_trace_if_available', action="store_true",
                        help='Whether to use previously stored simulations.')
    parser.add_argument('--ABC_alg', type=str, default="SABC",
                        help="ABC algorithm to use; can be PMCABC, APMCABC, SABC or RejectionABC")
    parser.add_argument('--ABC_steps', type=int, default=3, help="Number of steps for sequential ABC algorithms.")
    parser.add_argument('--ABC_full_output', action="store_true",
                        help="Whether to return full output in journal files.")
    parser.add_argument('--ABC_eps', type=float, default=None,
                        help="Provided numerical epsilon value to be passed to the ABC_inference function if ABC "
                             "inference is required. If provided, then epsilon is not automatically determined by "
                             "generating data from the prior and using epsilon_quantile.")
    parser.add_argument('--no_weighted_eucl_dist', action="store_true",
                        help="Disable rescaling in defining the distance for ABC.")
    parser.add_argument('--SMCABC_use_robust_kernel', action="store_true",
                        help="Use robust kernel by Lee (2012) in SMCABC. Useless for other ABC algorithms.")
    parser.add_argument('--SABC_cutoff', type=float, default=0.1)
    parser.add_argument('--root_folder', type=str, default=default_root_folder)
    parser.add_argument('--nets_folder', type=str, default=default_nets_folder)
    parser.add_argument('--observation_folder', type=str, default=default_observation_folder)
    parser.add_argument('--inference_folder', type=str, default=default_inference_folder)
    parser.add_argument('--no_bn', action="store_true")
    parser.add_argument('--affine_bn', action="store_true")
    parser.add_argument('--subsample_size', type=int, default=1000,
                        help='Number of samples used for Wass dist estimation.')
    parser.add_argument('--use_MPI', action="store_true", help='To perform ABC inference with MPI')
    parser.add_argument('--plot_marginal', action="store_true", help='Generate marginal plots (from MCMC samples).')
    parser.add_argument('--plot_bivariate', action="store_true", help='Generate bivariate plots (from MCMC samples).')
    parser.add_argument('--plot_trace', action="store_true", help='Generate trace plot (for exchange MCMC only).')
    parser.add_argument('--plot_journal', action="store_true",
                        help='Generate posterior plot from ABCpy journal (for ABC only).')
    parser.add_argument('--epsilon_quantile', type=float, default=0.03, help="The quantile value used to determine "
                                                                             "epsilon passed to the ABC_inference "
                                                                             "routine. Default is 0.03")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--debug_level', type=str,
                        help="Debug level to be used for loggin; can be 'info', 'debug', 'warn' "
                             "or 'error'; if none of those is given, 'warn' is assumed.")
    parser.add_argument('--tuning_window_exchange_MCMC', type=int, default=100,
                        help='Size of tuning window for ExcMCMC.')
    parser.add_argument('--bridging_exch_MCMC', type=int, default=0, help='Number of bridging steps for ExcMCMC.')
    parser.add_argument('--propose_new_theta_exchange_MCMC', type=str, help="Can be norm, truncnorm or transformation.",
                        default="transformation")
    return parser


def parser_plot_stats(default_root_folder, default_nets_folder="net-SM"):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help="The statistical model to consider.")
    parser.add_argument('--root_folder', type=str, default=default_root_folder)
    parser.add_argument('--nets_folder', type=str, default=default_nets_folder)
    parser.add_argument('--sleep', type=float, default=0, help='Minutes to sleep before starting')
    parser.add_argument('--n_observations', type=int, default=1000, help='Total number of observations.')
    parser.add_argument('--no_bn', action="store_true", help="Disable batch norm")
    parser.add_argument('--affine_bn', action="store_true")
    parser.add_argument('--seed', type=int, default=42, help='')

    return parser


def parser_SL_RE():
    parser = argparse.ArgumentParser()
    parser.add_argument('technique', help="Can be 'SL' (Synthetic Likelihood) or 'RE' (Ratio Estimation). ")
    parser.add_argument('model',
                        help="The statistical model to consider; can be 'AR2', 'MA2', 'beta', 'gamma', 'gaussian'")
    parser.add_argument('--sleep', type=float, default=0, help='Minutes to sleep before starting')
    parser.add_argument('--start_observation_index', type=int, default=0, help='Index to start from')
    parser.add_argument('--n_observations', type=int, default=10, help='Total number of observations.')
    parser.add_argument('--root_folder', type=str, default=None)
    parser.add_argument('--observation_folder', type=str, default="observations")
    parser.add_argument('--inference_folder', type=str, default="PMC-SL")
    parser.add_argument('--use_MPI', '-m', action="store_true")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--n_samples_per_param', type=int, default=100)
    parser.add_argument('--full_output', action="store_true", default="Whether to return full output in journal files.")
    parser.add_argument('--load_trace_if_available', action="store_true",
                        help='Whether to use previously stored simulations.')
    parser.add_argument('--subsample_size', type=int, default=1000,
                        help='Number of samples used for bivariate plots and Wass distance(if required).')
    parser.add_argument('--plot_marginal', action="store_true", help='Generate marginal plots (from MCMC samples).')
    parser.add_argument('--plot_bivariate', action="store_true", help='Generate bivariate plots (from MCMC samples).')
    parser.add_argument('--postprocessing', action="store_true",
                        help='Whether to compute metrics of performance of approx. posterior.')
    return parser

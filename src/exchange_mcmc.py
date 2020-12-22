import logging

import numpy as np
import torch
from scipy.stats import truncnorm
from tqdm import tqdm

from src.Transformations import TwoSidedBoundedVarScaler
from src.functions import DummyScaler, scale_thetas, scale_samples


def tune(scale, acc_rate):
    # THIS IS TAKEN FROM PYMC3 SOURCE CODE
    """Tunes the scaling parameter for the proposal
    distribution according to the acceptance rate over the
    last tune_interval:

    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10
    """
    if acc_rate < 0.001:
        # reduce by 90 percent
        scale *= 0.1
    elif acc_rate < 0.05:
        # reduce by 50 percent
        scale *= 0.5
    elif acc_rate < 0.2:
        # reduce by 10 percent
        scale *= 0.9
    elif acc_rate > 0.5:
        # increase by 10 percent
        scale *= 1.1
    elif acc_rate > 0.75:
        # increase by one hundred percent
        scale *= 2
    elif acc_rate > 0.95:
        # increase by one thousand percent
        scale *= 10

    return scale


def exchange_MCMC_with_SM_statistics(x_obs, initial_theta, prior_theta, net_data, net_theta=None, scaler_data=None,
                                     scaler_theta=None, propose_new_theta=None, T=30000, burn_in=1000,
                                     tuning_window_size=100, aux_MCMC_inner_steps=0,
                                     aux_MCMC_proposal_size=1, propose_new_aux_data=None, K=0, seed=None,
                                     debug_level=logging.WARN, lower_bounds_theta=None, upper_bounds_theta=None,
                                     **kwargs_propose_new_theta):
    """Implements exchange MCMC algorithm after learning statistics and parametrization with SM. It assumes that the
    last component of the output of `net_data` learns the c(x) term.

    initial_theta has to be a 1D array.

    If net_theta is None, we do not transform that and use the standard parameters that are provided.

    propose_new_theta, propose_new_aux_data, generate_sample, prior_theta are functions.

    If generate_sample is None: we do not start the inner MCMC from generating a sample from the true model.

    """
    logging.basicConfig(level=debug_level, filemode="w")

    logging.info(f"Prop. size inner MCMC {aux_MCMC_proposal_size}")
    logging.info(f"Inner MCMC steps {aux_MCMC_inner_steps}")

    # initialize stuff
    acc = 0
    acc_burnin = 0
    acc_tuning_window = 0
    accept_inner = 0
    accept_inner_before_burnin = 0
    accept_inner_before_burnin_tuning_window = 0
    accept_bridging = 0
    accept_bridging_before_burnin = 0
    n_proposed_theta_out_prior_range_burnin = 0
    n_proposed_theta_out_prior_range_window = 0
    n_proposed_theta_out_prior_range = 0
    size_theta = initial_theta.shape[0]
    trace = np.zeros((T + burn_in, size_theta))
    rng = np.random.RandomState(seed)
    old_theta = initial_theta  # set the initial value for theta

    if scaler_data is None:
        scaler_data = DummyScaler()
    if scaler_theta is None:
        scaler_theta = DummyScaler()

    if propose_new_theta is None:
        propose_new_theta_fcn = gaussian_perturbation_kernel
    # propose_new_theta can be a function or string
    elif isinstance(propose_new_theta, str):
        if propose_new_theta == "truncnorm":
            truncated_normal_kernel = TruncnormPerturbationKernel(lower_bounds_theta, upper_bounds_theta)
            propose_new_theta_fcn = truncated_normal_kernel.sample
            sigma = kwargs_propose_new_theta["sigma"]
        elif propose_new_theta == "norm":
            propose_new_theta_fcn = gaussian_perturbation_kernel
            sigma = kwargs_propose_new_theta["sigma"]
        elif propose_new_theta == "transformation":
            scaler_theta_proposal = TwoSidedBoundedVarScaler(
                lower_bound=lower_bounds_theta, upper_bound=upper_bounds_theta, rescale_transformed_vars=False).fit(
                initial_theta.reshape(1, -1))
            propose_new_theta_fcn = gaussian_perturbation_kernel
            log_det_jac_old_theta = scaler_theta_proposal.jac_log_det(old_theta)
            sigma = kwargs_propose_new_theta["sigma"]
        elif propose_new_theta == "adaptive":
            # from Haario et al. (2001)
            propose_new_theta_fcn = multivariate_normal_perturbation_kernel
            cov = np.diag(kwargs_propose_new_theta["sigma"])  # initial covariance values
            epsilon = 0.001
            t_0 = 10  # initial number of iterations with starting covariance matrix
            s_d = 2.4 ** 2 / size_theta  # scaling parameter
        elif propose_new_theta == "adaptive_transformation":
            # can also use the adaptive proposal on the transformed parameter space:
            scaler_theta_proposal = TwoSidedBoundedVarScaler(
                lower_bound=lower_bounds_theta, upper_bound=upper_bounds_theta, rescale_transformed_vars=False).fit(
                initial_theta.reshape(1, -1))
            log_det_jac_old_theta = scaler_theta_proposal.jac_log_det(old_theta)
            # from Haario et al. (2001)
            propose_new_theta_fcn = multivariate_normal_perturbation_kernel
            cov = np.diag(kwargs_propose_new_theta["sigma"])  # initial covariance values
            epsilon = 0.001
            t_0 = 10  # initial number of iterations with starting covariance matrix
            s_d = 2.4 ** 2 / size_theta  # scaling parameter
        else:
            raise NotImplementedError
    elif callable(propose_new_theta):
        propose_new_theta_fcn = propose_new_theta
    else:
        raise NotImplementedError

    if net_theta is not None:
        g_old_theta = torch.cat((net_theta(scale_thetas(scaler_theta, old_theta.reshape(1, -1)))[0], torch.ones(1)),
                                dim=0)
    else:
        g_old_theta = torch.cat((scale_thetas(scaler_theta, old_theta.reshape(1, -1))[0], torch.ones(1)),
                                dim=0)
    scaled_obs = scale_samples(scaler_data, x_obs, requires_grad=False)
    t_observation = net_data(scaled_obs).squeeze()

    with torch.no_grad():  # neglect grad computations; it gives small speedup.
        # outer loop (exchange MCMC)
        for t in tqdm(range(T + burn_in)):
            logging.debug(f"Outer loop iteration {t}")
            # propose new theta
            if propose_new_theta in ("transformation", "adaptive_transformation"):
                # need to propose a new theta with the transformation:
                # 1- transform old_theta
                old_theta_transformed = scaler_theta_proposal.transform(old_theta.reshape(1, -1)).reshape(-1)
                # 2- propose new_theta
                if propose_new_theta == "adaptive_transformation":
                    new_theta_transformed = propose_new_theta_fcn(old_theta_transformed, rng=rng, cov=cov)
                else:
                    new_theta_transformed = propose_new_theta_fcn(old_theta_transformed, rng=rng, sigma=sigma)
                # 3- transform that back
                new_theta = scaler_theta_proposal.inverse_transform(new_theta_transformed.reshape(1, -1)).reshape(-1)
            else:
                if propose_new_theta == "adaptive":
                    new_theta = propose_new_theta_fcn(old_theta, rng=rng, cov=cov)
                else:
                    new_theta = propose_new_theta_fcn(old_theta, rng=rng, sigma=sigma)
            # if the proposed new_theta is out of the region in which the prior is diff from 0, reject it immediately
            # Not sure this is great however, as sometimes the chain is stuck at the boundary.
            if prior_theta(new_theta) == 0:
                trace[t] = old_theta
                logging.info("Proposal out of parameter range at outer iteration {}".format(t))
                if t < burn_in:
                    n_proposed_theta_out_prior_range_burnin += 1
                    n_proposed_theta_out_prior_range_window += 1
                else:
                    n_proposed_theta_out_prior_range += 1
                continue
            if net_theta is not None:
                g_new_theta = torch.cat(
                    (net_theta(scale_thetas(scaler_theta, new_theta.reshape(1, -1)))[0], torch.ones(1)),
                    dim=0)
            else:
                g_new_theta = torch.cat((scale_thetas(scaler_theta, new_theta.reshape(1, -1))[0], torch.ones(1)),
                                        dim=0)

            # Liang (2010) proposal: start the inner MCMC from the observation value.
            aux_data_transformed_initial = scale_samples(scaler_data, x_obs, requires_grad=False)
            t_aux_data_initial = net_data(aux_data_transformed_initial).squeeze()
            # set the starting value of the inner chain to the initial value:
            aux_data_transformed = aux_data_transformed_initial
            t_aux_data = t_aux_data_initial

            if hasattr(scaler_data, "jac_log_det_inverse_transform"):
                # compute the jacobian for the inner MCMC if needed; this is sometimes repeated but it does not matter
                log_det_jac_old = scaler_data.jac_log_det_inverse_transform(aux_data_transformed)

            # now we run MCMC chain on the auxiliary data to target the right likelihood; we run MCMC on the transformed
            # data
            accept_inner_single_chain = 0
            for i in range(aux_MCMC_inner_steps):
                logging.debug(f"Inner loop iteration {i}")
                # propose new element. NOTE: I am proposing on the transformed data space, so that I need to correct for
                # that in the acceptance rate (by multiplying with the Jacobian) whenever the transformation I am using
                # is not linear.
                aux_data_transformed_proposal = gaussian_perturbation_kernel(aux_data_transformed,
                                                                             rng=rng, sigma=aux_MCMC_proposal_size)
                t_aux_data_proposal = net_data(aux_data_transformed_proposal.float()).squeeze()
                # compute acceptance rate; assume the proposal is symmetric for now.
                if hasattr(scaler_data, "jac_log_det_inverse_transform"):
                    log_det_jac_new = scaler_data.jac_log_det_inverse_transform(aux_data_transformed_proposal)
                    # log_det_jac_old = scaler_data.jac_log_det_inverse_transform(aux_data_transformed)
                    log_det_jac_diff = log_det_jac_old - log_det_jac_new
                else:
                    log_det_jac_diff = 0
                # compute acceptance; we use symmetric proposal
                alpha_inner = torch.exp(
                    torch.dot(t_aux_data_proposal - t_aux_data, g_new_theta) + log_det_jac_diff).item()
                # accept/reject
                if rng.uniform() < alpha_inner:
                    logging.debug(
                        "Inner MCMC: accepted proposal with mean {:.4f}".format(aux_data_transformed_proposal.mean()))
                    aux_data_transformed = aux_data_transformed_proposal
                    t_aux_data = t_aux_data_proposal
                    accept_inner_single_chain += 1
                    if hasattr(scaler_data, "jac_log_det_inverse_transform"):
                        log_det_jac_old = log_det_jac_new
            if aux_MCMC_inner_steps > 0:
                logging.info("inner acceptance rate at outer step {} was {:.4f}".format(
                    t, accept_inner_single_chain / aux_MCMC_inner_steps))

            # update the overall tracker for inner acceptance
            if t >= burn_in:  # we save the acceptance rate after burnin only.
                accept_inner += accept_inner_single_chain
            else:
                accept_inner_before_burnin += accept_inner_single_chain
                accept_inner_before_burnin_tuning_window += accept_inner_single_chain

            # compute acceptance rate of outer chain
            if propose_new_theta == "truncnorm":  # in this case proposal is not symmetric
                log_proposal_term = truncated_normal_kernel.log_pdf(new_theta, old_theta, sigma=sigma) \
                                    - truncated_normal_kernel.log_pdf(old_theta, new_theta, sigma=sigma)
            elif propose_new_theta in ("transformation", "adaptive_transformation"):
                # add new term taking into account the jacobian of the transformation; it is not strictly a log_proposal
                # term, but it can be thought of in that term too
                log_det_jac_new_theta = scaler_theta_proposal.jac_log_det(new_theta)
                # log_det_jac_old_theta = scaler_theta_proposal.jac_log_det(old_theta)
                log_proposal_term = log_det_jac_old_theta - log_det_jac_new_theta
            else:
                log_proposal_term = 0

            # if K == 0:
            # alpha = prior_theta(new_theta) / prior_theta(old_theta) * torch.exp(
            #         log_proposal_term + torch.dot(t_aux_data - t_observation, g_old_theta - g_new_theta))
            # put .item() otherwise it may be wrong to obtain a float instead of a tensor
            log_bridging_term = torch.dot(t_aux_data, g_old_theta - g_new_theta).item()
            # leave 0 + otherwise it may copy by reference the tensor/array -> make mess
            log_bridging_term_total = 0 + log_bridging_term
            intermediate_old_transformed = aux_data_transformed
            t_intermediate_old = t_aux_data

            acc_bridging_single = 0
            for k in range(1, K + 1):
                # apply the one-step MH transition kernel with the intermediate target density
                beta_k = (K - k + 1) / (K + 1)

                # propose intermediate x with a simple gaussian proposal:
                intermediate_proposal_transf = gaussian_perturbation_kernel(intermediate_old_transformed,
                                                                            rng=rng, sigma=aux_MCMC_proposal_size)
                t_intermediate_proposal = net_data(intermediate_proposal_transf.float()).squeeze()

                # we may be computing this on the transformed space as well: compute the jacobian
                if hasattr(scaler_data, "jac_log_det_inverse_transform"):
                    log_det_jac_new = scaler_data.jac_log_det_inverse_transform(intermediate_proposal_transf)
                    log_det_jac_diff = log_det_jac_old - log_det_jac_new
                else:
                    log_det_jac_diff = 0

                alpha_bridge = torch.exp(
                    log_det_jac_diff + torch.dot(beta_k * g_new_theta + (1 - beta_k) * g_old_theta,
                                                 t_intermediate_proposal - t_intermediate_old)).item()

                # draw uniform and save results
                if rng.uniform() < alpha_bridge:
                    logging.debug("Bridging at step {}: accepted proposal".format(k))
                    intermediate_old_transformed = intermediate_proposal_transf
                    t_intermediate_old = t_intermediate_proposal
                    acc_bridging_single += 1
                    log_bridging_term = torch.dot(t_intermediate_old, g_old_theta - g_new_theta).item()
                    if hasattr(scaler_data, "jac_log_det_inverse_transform"):
                        log_det_jac_old = log_det_jac_new

                # add to the log of the acceptance rate to be used in the following:
                # this can also be optimized easily
                log_bridging_term_total += log_bridging_term
                # log_bridging_term += torch.dot(t_intermediate_old, g_old_theta - g_new_theta)

            log_bridging_term_total /= (K + 1)
            if K > 0:
                logging.info(
                    "Acceptance rate for bridging at outer step {} was {:.4f}".format(t, acc_bridging_single / K))
            # update the overall tracker for inner acceptance
            if t >= burn_in:  # we save the acceptance rate after burnin only.
                accept_bridging += acc_bridging_single
            else:
                accept_bridging_before_burnin += acc_bridging_single

            alpha = prior_theta(new_theta) / prior_theta(old_theta) * np.exp(
                log_proposal_term + torch.dot(t_observation, g_new_theta - g_old_theta) + log_bridging_term_total)

            # draw uniform and save results
            if rng.uniform() < alpha:
                logging.info("Outer MCMC at step {}: accepted proposal".format(t))
                old_theta = new_theta
                g_old_theta = g_new_theta
                if propose_new_theta == "transformation":
                    log_det_jac_old_theta = log_det_jac_new_theta
                if t >= burn_in:  # we save the acceptance rate after burnin only.
                    acc += 1
                else:
                    acc_burnin += 1
                    acc_tuning_window += 1
                # update the initial value of the inner MCMC:
                # aux_data_transformed_initial = aux_data_transformed
                # t_aux_data_initial = t_aux_data

            trace[t] = old_theta
            if propose_new_theta in ("adaptive", "adaptive_transformation") and t > t_0:
                # update here the adaptive covariance following Haario recursion formula
                # (cheaper than computing full cov from trace):
                sum_up_t = np.mean(trace[:(t + 1)], axis=0)
                sum_up_t_minus_one = np.mean(trace[:t], axis=0)
                cov = ((t - 1) * cov + s_d * (
                        t * np.einsum('i,j->ij', sum_up_t_minus_one, sum_up_t_minus_one) - (t + 1) * np.einsum(
                    'i,j->ij', sum_up_t, sum_up_t) + np.einsum('i,j->ij', old_theta, old_theta) + epsilon * np.eye(
                    size_theta))) / t

            if t < burn_in and (t + 1) % tuning_window_size == 0 and propose_new_theta in (
                    "transformation", "norm", "truncnorm"):
                # tune the proposal size here:
                # here we actually need to keep trace of acc rate in the last tuning window.
                sigma = tune(sigma, acc_tuning_window / tuning_window_size)
                logging.info(f"tune at step {t} with window acc rate {acc_tuning_window / tuning_window_size:.4f}")
                # is the above correct or should I also take into account the number of proposals out of prior range?
                # this does not matter if transformation or truncnorm proposals are used in outer chain.
                acc_tuning_window = 0
                # We also tune the proposal size of the inner chain; start by tuning the scalar proposal size each
                # tuning_window_size outer iterations, according to acc rate.
                if aux_MCMC_inner_steps > 0:
                    acc_rate_inner_window = accept_inner_before_burnin_tuning_window / (
                            aux_MCMC_inner_steps * (tuning_window_size - n_proposed_theta_out_prior_range_window))
                    aux_MCMC_proposal_size = tune(aux_MCMC_proposal_size, acc_rate_inner_window)
                    logging.info(f"tune at step {t} with window inner acc rate {acc_rate_inner_window:.4f}")
                accept_inner_before_burnin_tuning_window = 0
                n_proposed_theta_out_prior_range_window = 0
                # burned_in_trace = trace[burn_in:]
    if T > 0:
        print("\nOverall acceptance rate was {:.4f}".format(acc / T))
    if T - n_proposed_theta_out_prior_range > 0:
        print("\nAcceptance rate excluding proposals out of prior range {:.4f}".format(
            acc / (T - n_proposed_theta_out_prior_range)))
    if T > 0:
        print("\nRatio of ExchangeMCMC steps for which proposal was out of prior range {:.4f}".format(
            n_proposed_theta_out_prior_range / T))
    if burn_in > 0:
        print("\nOverall acceptance rate during burn-in was {:.4f}".format(acc_burnin / burn_in))
    if T - n_proposed_theta_out_prior_range > 0:
        print("\nAcceptance rate excluding proposals out of prior range during burn-in {:.4f}".format(
            acc_burnin / (burn_in - n_proposed_theta_out_prior_range_burnin)))
    if burn_in > 0:
        print("\nRatio of ExchangeMCMC steps for which proposal was out of prior range during burn-n {:.4f}".format(
            n_proposed_theta_out_prior_range_burnin / burn_in))
    if (T - n_proposed_theta_out_prior_range) * aux_MCMC_inner_steps > 0:
        print("\nOverall inner acceptance rate was {:.4f}".format(
            accept_inner / ((T - n_proposed_theta_out_prior_range) * aux_MCMC_inner_steps)))
    if (burn_in - n_proposed_theta_out_prior_range_burnin) * aux_MCMC_inner_steps > 0:
        print("\nInner acceptance rate during burn-in was {:.4f}\n".format(
            accept_inner_before_burnin / ((burn_in - n_proposed_theta_out_prior_range_burnin) * aux_MCMC_inner_steps)))
    if (T - n_proposed_theta_out_prior_range) * K > 0:
        print("\nOverall bridging acceptance rate was {:.4f}".format(
            accept_bridging / ((T - n_proposed_theta_out_prior_range) * K)))
    if (burn_in - n_proposed_theta_out_prior_range_burnin) * K > 0:
        print("\nInner bridging rate during burn-in was {:.4f}\n".format(
            accept_bridging_before_burnin / ((burn_in - n_proposed_theta_out_prior_range_burnin) * K)))

    return trace


def gaussian_perturbation_kernel(old_position, rng, sigma=1):
    """Implement a new proposal with independent Gaussian components, with standard deviation sigma.
    Sigma can be either a scalar, which is taken to be the std of all the components of theta, or an
    array with same size as theta."""
    return old_position + rng.normal(scale=sigma, size=old_position.shape)


def multivariate_normal_perturbation_kernel(old_position, rng, cov=None):
    """Implement a new proposal with multivariate normal."""
    if cov is None:
        cov = np.eye(old_position.shape[0])
    return rng.multivariate_normal(mean=old_position, cov=cov)


class TruncnormPerturbationKernel():

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def sample(self, old_position, rng, sigma):
        # iteratively propose until we get a sample which is in the correct region; I think this is what
        # scipy.stats.truncnorm does internally, but harder to use
        ok = False
        while not ok:
            new_position = old_position + rng.normal(scale=sigma, size=old_position.shape)
            ok = np.logical_and(new_position > self.lower_bound, new_position < self.upper_bound).all()

        return new_position

    def log_pdf(self, old_position, new_position, sigma):
        # Evaluate the log pdf of the truncated normal proposal at the new position with mean old_position and std dev
        # sigma
        if not hasattr(sigma, "__len__"):
            sigma = np.ones_like(old_position) * sigma
        old_position = [old_position] if not hasattr(old_position, "__len__") else old_position
        new_position = [new_position] if not hasattr(new_position, "__len__") else new_position
        sigma = [sigma] if not hasattr(sigma, "__len__") else sigma

        tot_log_pdf = 0
        for i in range(len(old_position)):
            a, b = (self.lower_bound[i] - old_position[i]) / sigma[i], (self.upper_bound[i] - old_position[i]) / sigma[
                i]
            tot_log_pdf += truncnorm.logpdf(new_position[i], a, b, loc=old_position[i], scale=sigma[i])
        return tot_log_pdf


def uniform_prior_theta(theta, lower_bounds, upper_bounds):
    if (lower_bounds < theta).all() and (theta < upper_bounds).all():
        return 1
    else:
        return 0

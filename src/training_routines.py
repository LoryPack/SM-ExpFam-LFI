import numpy as np
import torch
from abcpy.NN_utilities.utilities import save_net
from torch.optim import Adam
from tqdm import tqdm

from src.functions import jacobian_second_order
from src.functions import set_requires_grad
from src.losses import Fisher_divergence_loss, Fisher_divergence_loss_with_c_x
from src.networks import IdentityNet


def _base_training_routine(samples_matrix, theta_vect, batch_steps, net, net_theta,
                           samples_matrix_test=None, theta_vect_test=None, epochs_test_interval=10,
                           epochs_before_early_stopping=100,
                           lambda_l2=0,
                           early_stopping=False,
                           n_epochs=100, batch_size=None, lr=0.001, lr_theta=0.001, seed=None,
                           return_stat_list=False, return_nat_par_list=False, return_loss_list=False,
                           return_test_loss_list=False,
                           enable_scheduler=True, with_c_x=False, cuda=False, load_all_data_GPU=False,
                           return_final_lr=False, return_final_lr_theta=False,
                           update_batchnorm_running_means_before_eval=False,
                           save_net_at_each_epoch=False, net_folder_path=None,
                           **kwargs_batch_steps):
    """This assumes samples matrix to be a 2d tensor with size (n_theta, size_sample) and theta_vect a 2d tensor with
    size (n_theta, p).
    """
    if save_net_at_each_epoch and net_folder_path is None:
        raise RuntimeError("Need to provide path where to save the net if you want to save them at each iterations.")
    # set up cuda:
    device = "cuda" if cuda and torch.cuda.is_available() else "cpu"

    net.to(device)
    net_theta.to(device)

    if load_all_data_GPU:
        # we move all data to the gpu; it needs to be small enough
        samples_matrix = samples_matrix.to(device)
        if samples_matrix_test is not None:
            samples_matrix_test = samples_matrix_test.to(device)
        theta_vect = theta_vect.to(device)
        if theta_vect_test is not None:
            theta_vect_test = theta_vect_test.to(device)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    compute_test_loss = False
    if theta_vect_test is not None and samples_matrix_test is not None:
        test_loss_list = []
        compute_test_loss = True
        n_theta_test = theta_vect_test.shape[0]

    # TODO pass different optimizers instead of the default ones.

    sgd = Adam(net.parameters(), lr=lr, weight_decay=lambda_l2)
    sgd_theta = Adam(net_theta.parameters(), lr=lr_theta, weight_decay=lambda_l2)

    if batch_size is None:  # in this case use full batch
        batch_size = theta_vect.shape[0]

    n_theta = theta_vect.shape[0]

    if return_nat_par_list:
        learned_nat_param_list = np.zeros((n_epochs, n_theta, theta_vect.shape[-1]))
    if return_stat_list:
        # learned_stat_list = np.zeros((n_epochs, n_theta, theta_vect.shape[-1] + (1 if with_c_x else 0)))
        learned_stat_list = np.zeros((n_epochs, batch_size, theta_vect.shape[-1] + (1 if with_c_x else 0)))
    loss_list = []

    # define now the LR scheduler: 

    lr_scheduler_class_name = 'ExponentialLR'
    lr_scheduler_kwargs = dict(gamma=0.99)

    scheduler = torch.optim.lr_scheduler.__dict__[lr_scheduler_class_name](sgd, **lr_scheduler_kwargs)
    scheduler_theta = torch.optim.lr_scheduler.__dict__[lr_scheduler_class_name](sgd_theta, **lr_scheduler_kwargs)

    # initialize the state_dict variables:
    net_state_dict = None
    net_state_dict_theta = None

    for epoch in tqdm(range(n_epochs)):
        # print("epoch", epoch)

        # set nets to train mode:
        net.train()
        net_theta.train()

        indeces = np.random.permutation(n_theta)  # this may be a bottleneck computationally?
        batch_index = 0
        total_train_loss_epoch = 0

        # loop over batches
        while batch_size * batch_index < n_theta:

            sgd.zero_grad()
            sgd_theta.zero_grad()

            # by writing in this way, if we go above the number of elements in the vector, you don't care
            batch_indeces = indeces[batch_size * batch_index:batch_size * (batch_index + 1)]

            thetas_batch = theta_vect[batch_indeces].to(device)

            # compute the transformed parameter values for the batch:
            etas = net_theta(thetas_batch)

            samples_batch = samples_matrix[batch_indeces].to(device)
            # now call the batch routine that takes care of forward step of simulations as well
            batch_loss = batch_steps(net, samples_batch, etas, **kwargs_batch_steps)

            total_train_loss_epoch += batch_loss.item()

            # set requires_grad to False to save computation
            if lr == 0:
                set_requires_grad(net, False)
            if lr_theta == 0:
                set_requires_grad(net_theta, False)

            batch_loss.backward()

            # reset it
            if lr == 0:
                set_requires_grad(net, True)
            if lr_theta == 0:
                set_requires_grad(net_theta, True)

            sgd.step()
            sgd_theta.step()

            batch_index += 1

        if return_loss_list:
            loss_list.append(total_train_loss_epoch / (batch_index + 1))

        # at each epoch we compute the test loss; we need to use batches as well here, otherwise it may not fit to GPU
        # memory
        if compute_test_loss:
            # first, we do forward pass of all the training data in order to update the batchnorm running means (if a
            # batch norm layer is there):
            if update_batchnorm_running_means_before_eval:
                with torch.no_grad():
                    batch_index = 0
                    while batch_size * batch_index < n_theta:
                        # the batchnorm is usually after the net; then, it is enough to feedforward the data there:
                        thetas_batch = theta_vect[batch_size * batch_index:batch_size * (batch_index + 1)].to(
                            device)
                        _ = net_theta(thetas_batch)
                        batch_index += 1

            net.eval()
            net_theta.eval()

            batch_index = 0
            total_test_loss_epoch = 0
            while batch_size * batch_index < n_theta_test:
                # no need to shuffle the test data:
                thetas_batch = theta_vect_test[batch_size * batch_index:batch_size * (batch_index + 1)].to(device)
                samples_batch = samples_matrix_test[batch_size * batch_index:batch_size * (batch_index + 1)].to(device)

                # compute the transformed parameter values for the batch:
                etas_test = net_theta(thetas_batch)

                total_test_loss_epoch += batch_steps(net, samples_batch, etas_test, **kwargs_batch_steps).item()

                batch_index += 1

            test_loss_list.append(total_test_loss_epoch / (batch_index + 1))

            # the test loss on last step is larger than the training_dataset_index before, stop training
            if early_stopping and (epoch + 1) % epochs_test_interval == 0:
                # after `epochs_before_early_stopping` epochs, we can stop only if we saved a state_dict before
                # (ie if at least epochs_test_interval epochs have passed).
                if epoch + 1 > epochs_before_early_stopping and net_state_dict is not None:
                    if test_loss_list[-1] > test_loss_list[- 1 - epochs_test_interval]:
                        print("Training has been early stopped at epoch {}.".format(epoch + 1))
                        # reload the previous state dict:
                        net.load_state_dict(net_state_dict)
                        net_theta.load_state_dict(net_state_dict_theta)
                        break  # stop training
                # if we did not stop: update the state dict
                net_state_dict = net.state_dict()
                net_state_dict_theta = net_theta.state_dict()

        if return_stat_list:
            batch_index = 0
            # while batch_size * batch_index < n_theta_test:
            # save only `batch_size` statistics, otherwise they are too many!
            samples_batch = samples_matrix[batch_size * batch_index:batch_size * (batch_index + 1)].reshape(
                batch_size, -1).to(device)
            learned_stat_list[epoch, batch_size * batch_index:batch_size * (batch_index + 1)] = net(
                samples_batch).detach().cpu().numpy()
            # batch_index += 1
            # .reshape(-1, theta_vect.shape[-1] + (1 if with_c_x else 0))

        if return_nat_par_list:
            learned_nat_param_list[epoch] = net_theta(theta_vect).detach().numpy().reshape(-1, theta_vect.shape[-1])

        if enable_scheduler:
            scheduler.step()
            scheduler_theta.step()

        if save_net_at_each_epoch:
            save_net(net_folder_path + "net_data_SM.pth", net)
            save_net(net_folder_path + "net_theta_SM.pth", net_theta)
            # should add condition if we do FP:
            # save_net(net_folder_path + "net_data_FP.pth", net_data)
            # save losses:
            np.save(net_folder_path + "loss.npy", np.array(loss_list))
            if compute_test_loss:
                np.save(net_folder_path + "test_loss.npy", np.array(test_loss_list))

    # after training, return to eval mode:
    net.eval()
    net_theta.eval()

    # learned_statistics_test_list.append(net(samples_matrix_test).detach().numpy())

    return_arguments = []
    if return_nat_par_list:
        return_arguments.append(learned_nat_param_list)
    if return_stat_list:
        return_arguments.append(learned_stat_list)
    if return_loss_list:
        return_arguments.append(loss_list)
    if return_test_loss_list and compute_test_loss:
        return_arguments.append(test_loss_list)
    # now return the final learning rates; in this way, if we are alternating the training, we don't lose the scheduler
    # updates
    if return_final_lr:
        return_arguments.append(sgd.state_dict()["param_groups"][0]["lr"])
    if return_final_lr_theta:
        return_arguments.append(sgd_theta.state_dict()["param_groups"][0]["lr"])

    # move back the nets to cpu:
    net.cpu()
    net_theta.cpu()

    if len(return_arguments) > 0:
        return return_arguments


# ALL POSSIBLE VARIANTS OF FISHER DIVERGENCE:

def batch_Fisher_div(net, samples, etas):
    # do the forward pass at once here:
    transformed_samples = net(samples)

    f, s = jacobian_second_order(samples, transformed_samples)
    # we reshape the f and s
    f = f.reshape(-1, f.shape[1], f.shape[2])
    s = s.reshape(-1, s.shape[1], s.shape[2])

    return Fisher_divergence_loss(f, s, etas) / (samples.shape[0])


def batch_Fisher_div_with_c_x(net, samples, etas, lam=0):
    # do the forward pass at once here:
    transformed_samples = net(samples)

    f, s = jacobian_second_order(samples, transformed_samples)
    f = f.reshape(-1, f.shape[1], f.shape[2])
    s = s.reshape(-1, s.shape[1], s.shape[2])

    return Fisher_divergence_loss_with_c_x(f, s, etas, lam=lam) / (samples.shape[0])


# IMPLEMENTATIONS OF THE FISHER DIVERGENCE WORKING WITH NETWORKS THAT COMPUTE DERIVATIVES IN FORWARD PASS.

def batch_Fisher_div_net_with_derivatives(net, samples, etas, lam=0):
    """lam is the regularization parameter of the Kingm & LeCun (2010) regularization"""
    # samples here is a 3d tensor

    # do the forward pass and obtain derivatives at once here:
    transformed_samples, f, s = net.forward_and_derivatives(samples)

    # we reshape the f and s
    f = f.reshape(-1, f.shape[1], f.shape[2])
    s = s.reshape(-1, s.shape[1], s.shape[2])

    return Fisher_divergence_loss(f, s, etas, lam=lam) / (samples.shape[0])


def batch_Fisher_div_with_c_x_net_with_derivatives(net, samples, etas, lam=0):
    # do the forward pass and obtain derivatives at once here:
    transformed_samples, f, s = net.forward_and_derivatives(samples)

    f = f.reshape(-1, f.shape[1], f.shape[2])
    s = s.reshape(-1, s.shape[1], s.shape[2])

    return Fisher_divergence_loss_with_c_x(f, s, etas, lam=lam) / (samples.shape[0])


# L2 BATCH LEVEL AND TRAINING ROUTINE:

def batch_l2_loss(net, samples, etas):
    # now reshape to perform forward pass
    reshaped_samples = samples.view(-1, samples.shape[-1])

    # do the forward pass at once here:
    transformed_samples = net(reshaped_samples)

    return torch.nn.MSELoss()(transformed_samples, etas)


def FP_training_routine(samples_matrix, theta_vect, net,
                        samples_matrix_test=None, theta_vect_test=None, epochs_test_interval=10,
                        n_epochs=100, early_stopping=False,
                        batch_size=None, lr=0.001,
                        return_stat_list=False, return_nat_par_list=False, return_loss_list=False,
                        return_test_loss_list=False, seed=None, **kwargs):
    return _base_training_routine(samples_matrix, theta_vect, batch_l2_loss, net, net_theta=IdentityNet(),
                                  samples_matrix_test=samples_matrix_test, theta_vect_test=theta_vect_test,
                                  epochs_test_interval=epochs_test_interval,
                                  n_epochs=n_epochs, early_stopping=early_stopping,
                                  batch_size=batch_size, lr=lr, lr_theta=0,
                                  return_stat_list=return_stat_list, return_nat_par_list=return_nat_par_list,
                                  return_loss_list=return_loss_list, return_test_loss_list=return_test_loss_list,
                                  seed=seed, **kwargs)


# FISHER DIVERGENCE VARIANTS
def Fisher_divergence_training_routine(samples_matrix, theta_vect, net, net_theta,
                                       samples_matrix_test=None, theta_vect_test=None, epochs_test_interval=10,
                                       n_epochs=100, early_stopping=False,
                                       batch_size=None, lr=0.001, lr_theta=0.001,
                                       return_stat_list=False, return_nat_par_list=False, return_loss_list=False,
                                       return_test_loss_list=False, seed=None, **kwargs):
    return _base_training_routine(samples_matrix, theta_vect, batch_Fisher_div, net, net_theta,
                                  samples_matrix_test=samples_matrix_test, theta_vect_test=theta_vect_test,
                                  epochs_test_interval=epochs_test_interval,
                                  n_epochs=n_epochs, early_stopping=early_stopping,
                                  batch_size=batch_size, lr=lr, lr_theta=lr_theta,
                                  return_stat_list=return_stat_list, return_nat_par_list=return_nat_par_list,
                                  return_loss_list=return_loss_list, return_test_loss_list=return_test_loss_list,
                                  seed=seed, **kwargs)


def Fisher_divergence_training_routine_with_c_x(samples_matrix, theta_vect, net, net_theta,
                                                samples_matrix_test=None, theta_vect_test=None, epochs_test_interval=10,
                                                n_epochs=100, early_stopping=False,
                                                batch_size=None, lr=0.001, lr_theta=0.001,
                                                return_stat_list=False, return_nat_par_list=False,
                                                return_loss_list=False, return_test_loss_list=False,
                                                seed=None, **kwargs):
    return _base_training_routine(samples_matrix, theta_vect, batch_Fisher_div_with_c_x, net, net_theta,
                                  samples_matrix_test=samples_matrix_test, theta_vect_test=theta_vect_test,
                                  epochs_test_interval=epochs_test_interval,
                                  n_epochs=n_epochs, early_stopping=early_stopping,
                                  batch_size=batch_size, lr=lr, lr_theta=lr_theta,
                                  return_stat_list=return_stat_list, return_nat_par_list=return_nat_par_list,
                                  return_loss_list=return_loss_list, return_test_loss_list=return_test_loss_list,
                                  seed=seed, with_c_x=True, **kwargs)


# FISHER DIVERGENCE VARIANTS WORKING WITH NETWORKS THAT COMPUTE DERIVATIVES IN FORWARD PASS.

def Fisher_divergence_training_routine_net_with_derivatives(samples_matrix, theta_vect, net, net_theta,
                                                            samples_matrix_test=None, theta_vect_test=None,
                                                            epochs_test_interval=10,
                                                            n_epochs=100, early_stopping=False,
                                                            batch_size=None, lr=0.001, lr_theta=0.001,
                                                            return_stat_list=False, return_nat_par_list=False,
                                                            return_loss_list=False,
                                                            return_test_loss_list=False, seed=None,
                                                            **kwargs):
    return _base_training_routine(samples_matrix, theta_vect, batch_Fisher_div_net_with_derivatives, net, net_theta,
                                  samples_matrix_test=samples_matrix_test, theta_vect_test=theta_vect_test,
                                  epochs_test_interval=epochs_test_interval,
                                  n_epochs=n_epochs, early_stopping=early_stopping,
                                  batch_size=batch_size, lr=lr, lr_theta=lr_theta,
                                  return_stat_list=return_stat_list, return_nat_par_list=return_nat_par_list,
                                  return_loss_list=return_loss_list, return_test_loss_list=return_test_loss_list,
                                  seed=seed, **kwargs)


def Fisher_divergence_training_routine_with_c_x_net_with_derivatives(samples_matrix, theta_vect, net, net_theta,
                                                                     samples_matrix_test=None, theta_vect_test=None,
                                                                     epochs_test_interval=10,
                                                                     n_epochs=100, early_stopping=False,
                                                                     batch_size=None, lr=0.001, lr_theta=0.001,
                                                                     return_stat_list=False, return_nat_par_list=False,
                                                                     return_loss_list=False,
                                                                     return_test_loss_list=False,
                                                                     seed=None, **kwargs):
    return _base_training_routine(samples_matrix, theta_vect, batch_Fisher_div_with_c_x_net_with_derivatives, net,
                                  net_theta,
                                  samples_matrix_test=samples_matrix_test, theta_vect_test=theta_vect_test,
                                  epochs_test_interval=epochs_test_interval,
                                  n_epochs=n_epochs, early_stopping=early_stopping,
                                  batch_size=batch_size, lr=lr, lr_theta=lr_theta,
                                  return_stat_list=return_stat_list, return_nat_par_list=return_nat_par_list,
                                  return_loss_list=return_loss_list, return_test_loss_list=return_test_loss_list,
                                  seed=seed, with_c_x=True, **kwargs)

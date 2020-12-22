import torch


def Fisher_divergence_loss(first_der_t, second_der_t, eta, lam=0):
    """lam is the regularization parameter of the Kingm & LeCun (2010) regularization"""
    inner_prod_second_der_eta = torch.bmm(second_der_t, eta.unsqueeze(-1))  # this is used twice

    if lam == 0:
        return sum(
            (0.5 * torch.bmm(first_der_t, eta.unsqueeze(-1)) ** 2 + inner_prod_second_der_eta).view(-1))
    else:
        return sum(
            (0.5 * torch.bmm(first_der_t, eta.unsqueeze(-1)) ** 2 +
             inner_prod_second_der_eta + lam * inner_prod_second_der_eta ** 2).view(-1))


def Fisher_divergence_loss_with_c_x(first_der_t, second_der_t, eta, lam=0):
    # this enables to use the term c(x) in the approximating family, ie a term that depends only on x and not on theta.
    new_eta = torch.cat((eta, torch.ones(eta.shape[0], 1).to(eta)),
                        dim=1)  # the one tensor need to be on same device as eta.
    # then call the other loss function with this new_eta:
    return Fisher_divergence_loss(first_der_t, second_der_t, new_eta, lam=lam)
    # return sum((0.5 * torch.bmm(first_der_t, new_eta.unsqueeze(-1)) ** 2 + torch.bmm(second_der_t,
    #                                                                                  new_eta.unsqueeze(-1))).view(-1))
    # note: we use `sum` instead of `torch.sum` as the latter does not work in GPU (backward fails).

# from fml.functional import sinkhorn

from utils import *
from utils_largeea import *

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
import torch


def sinkhorn_norm(alpha: torch.Tensor, n_iter: int = 20) -> (torch.Tensor,):
    for _ in range(n_iter):
        alpha = alpha / alpha.sum(-1, keepdim=True)
        alpha = alpha / alpha.sum(-2, keepdim=True)
    return alpha


def log_sinkhorn_norm(log_alpha: torch.Tensor, n_iter: int = 100) -> (torch.Tensor,):
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
    return log_alpha.exp()


def gumbel_sinkhorn(log_alpha: torch.Tensor, tau: float = 1.0, n_iter: int = 20, noise: bool = True) -> (torch.Tensor,):
    if noise:
        uniform_noise = torch.rand_like(log_alpha)
        gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-20) + 1e-20)
        log_alpha = (log_alpha + gumbel_noise) / tau
    sampled_perm_mat = log_sinkhorn_norm(log_alpha, n_iter)
    return sampled_perm_mat


def gen_assignment(cost_matrix):
    row, col = linear_sum_assignment(cost_matrix)
    np_assignment_matrix = coo_matrix((np.ones_like(row), (row, col))).toarray()
    return np_assignment_matrix


def gumbel_matching(log_alpha: torch.Tensor, noise: bool = True) -> (torch.Tensor,):
    if noise:
        uniform_noise = torch.rand_like(log_alpha)
        gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-20) + 1e-20)
        log_alpha = (log_alpha + gumbel_noise)
    np_log_alpha = log_alpha.detach().to("cpu").numpy()
    np_assignment_matrices = [gen_assignment(-x) for x in np_log_alpha]
    np_assignment_matrices = np.stack(np_assignment_matrices, 0)
    assignment_matrices = torch.from_numpy(np_assignment_matrices).float().to(log_alpha.device)
    return assignment_matrices


def inverse_permutation(X, permutation_matrix):
    return torch.einsum("bpq,bp->bq", (permutation_matrix, X))


def inverse_permutation_for_image(X, permutation_matrix):
    return torch.einsum("bpq,bpchw->bqchw", (permutation_matrix, X)).contiguous()


def sinkhorn(a: torch.Tensor, b: torch.Tensor, M: torch.Tensor, eps: float,
             max_iters: int = 100, stop_thresh: float = 1e-3):
    """
    Compute the Sinkhorn divergence between two sum of dirac delta distributions, U, and V.
    This implementation is numerically stable with float32.
    :param a: A m-sized minibatch of weights for each dirac in the first distribution, U. i.e. shape = [m, n]
    :param b: A m-sized minibatch of weights for each dirac in the second distribution, V. i.e. shape = [m, n]
    :param M: A minibatch of n-by-n tensors storing the distance between each pair of diracs in U and V.
             i.e. shape = [m, n, n] and each i.e. M[k, i, j] = ||u[k,_i] - v[k, j]||
    :param eps: The reciprocal of the sinkhorn regularization parameter
    :param max_iters: The maximum number of Sinkhorn iterations
    :param stop_thresh: Stop if the change in iterates is below this value
    :return:
    """
    # a and b are tensors of size [m, n]
    # M is a tensor of size [m, n, n]

    nb = M.shape[0]
    m = M.shape[1]
    n = M.shape[2]

    if a.dtype != b.dtype or a.dtype != M.dtype:
        raise ValueError("Tensors a, b, and M must have the same dtype got: dtype(a) = %s, dtype(b) = %s, dtype(M) = %s"
                         % (str(a.dtype), str(b.dtype), str(M.dtype)))
    if a.device != b.device or a.device != M.device:
        raise ValueError("Tensors a, b, and M must be on the same device got: "
                         "device(a) = %s, device(b) = %s, device(M) = %s"
                         % (a.device, b.device, M.device))
    if len(M.shape) != 3:
        raise ValueError("Got unexpected shape for M (%s), should be [nb, m, n] where nb is batch size, and "
                         "m and n are the number of samples in the two input measures." % str(M.shape))
    if torch.Size(a.shape) != torch.Size([nb, m]):
        raise ValueError("Got unexpected shape for tensor a (%s). Expected [nb, m] where M has shape [nb, m, n]." %
                         str(a.shape))

    if torch.Size(b.shape) != torch.Size([nb, n]):
        raise ValueError("Got unexpected shape for tensor b (%s). Expected [nb, n] where M has shape [nb, m, n]." %
                         str(b.shape))

    # Initialize the iteration with the change of variable
    u = torch.zeros(a.shape, dtype=a.dtype, device=a.device)
    v = eps * torch.log(b)

    M_t = torch.transpose(M, 1, 2)

    def stabilized_log_sum_exp(x):
        # max_x = torch.max(x, dim=2)[0]
        # x = x - max_x.unsqueeze(2)
        # ret = torch.log(torch.sum(torch.exp(x), dim=2)) + max_x
        # return ret
        return torch.logsumexp(x, -1)

    for current_iter in range(max_iters):
        u_prev = u
        v_prev = v

        summand_u = (-M + v.unsqueeze(1)) / eps
        u = eps * (torch.log(a) - stabilized_log_sum_exp(summand_u))

        summand_v = (-M_t + u.unsqueeze(1)) / eps
        v = eps * (torch.log(b) - stabilized_log_sum_exp(summand_v))

        err_u = torch.max(torch.sum(torch.abs(u_prev - u), dim=1))
        err_v = torch.max(torch.sum(torch.abs(v_prev - v), dim=1))

        if err_u < stop_thresh and err_v < stop_thresh:
            break

    log_P = (-M + u.unsqueeze(2) + v.unsqueeze(1)) / eps

    P = torch.exp(log_P)

    return P


def matrix_sinkhorn(pred_or_m, expected=None, a=None, b=None, dist_func=cosine_distance, device='cuda'):
    if expected is None:
        M = view3(pred_or_m).to(torch.float32).to(device)
        m, n = tuple(pred_or_m.size())
    else:
        m = pred_or_m.size(0)
        n = expected.size(0)
        M = dist_func(pred_or_m.to(device), expected.to(device))
        M = view3(M)

    if a is None:
        a = torch.ones([1, m], requires_grad=False, device=device)
    else:
        a = a.to(device)

    if b is None:
        b = torch.ones([1, n], requires_grad=False, device=device)
    else:
        b = b.to(device)
    time0 = time.time()
    P = sinkhorn(a, b, M, 1e-3, max_iters=100, stop_thresh=1e-3)
    time1 = time.time()
    print('Sinkhorn time=', time1 - time0)
    return view2(P)

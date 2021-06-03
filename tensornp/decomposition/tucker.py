'''
@Author: duke
@Date: 2021-05-14 23:20:07
@LastEditTime: 2021-06-03 10:28:28
@LastEditors: duke
@Description: HOOI, High-order Orthogonal Iteration implementation of tucker decomposition
'''
from .. import backend as T
from .. import Tensor
from ..linalg import unfold, fold, norm
from .hosvd import reconstruct_hosvd
from . import hosvd


reconstruct_tucker = reconstruct_hosvd


def tucker(tensor: Tensor, ranks: list, stop_iter=0, tol=1e-6, verbose=0):
    """High-order Orthogonal Iteration, Tucker 2 decomposition.

    Args:
        tensor (Tensor): Array-like tensor to decomposition.
        ranks (list): List of rank, specify size of every mode component. Should have size equals to order of tensor.
    """ 
    order = tensor.ndim
    factors = hosvd(tensor, ranks, compute_core=False)
    shape = list(tensor.shape)
    error_list = []
    
    # ALS optimization
    for iteration in range(stop_iter):
        for i in range(order):
            gamma = tensor
            for i1 in range(order):
                if i1 == i:
                    continue
                shape[i1] = factors[i1].shape[1]
                gamma = fold(factors[i1].T @ unfold(gamma, i1), i1, shape)
            u, _, _ = T.svd(unfold(gamma, i), full_matrices=False)
            factors[i] = u[:, :ranks[i]]

        if tol:
            g = tensor
            for i in range(order):
                shape[i] = factors[i].shape[1]
                g = fold(factors[i].T @ unfold(g, i), i, shape)
            rec_tensor = reconstruct_tucker(g, factors)
            norm_error = norm(rec_tensor - tensor) / norm(tensor)
            error_list.append(norm_error)
            if iteration >= 1:
                error_decrease = error_list[-2] - error_list[-1]
                if verbose > 1:
                    print("iteration [{}] norm error: {:.5f} | decrease = {:7.2e}".
                        format(iteration, norm_error, error_decrease))
                if error_decrease < tol:
                    if verbose:
                        print("HOOI-ALS converged after {} iterations".format(iteration))
                    return g, factors
            else:
                if verbose > 1:
                    print("iteration [{}] norm error: {:.5f}".
                        format(iteration, norm_error))
        g = tensor
        for i in range(order):
            shape[i] = factors[i].shape[1]
            g = fold(factors[i].T @ unfold(g, i), i, shape)
            
    return g, factors
    


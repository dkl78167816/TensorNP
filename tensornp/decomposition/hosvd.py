'''
@Author: duke
@Date: 2021-05-14 23:20:02
@LastEditTime: 2021-06-03 09:09:10
@LastEditors: duke
@Description: HOSVD, High Order Singular Value Decomposition
'''
from .. import backend as T
from .. import Tensor
from ..linalg import unfold, fold


def hosvd(tensor: Tensor, ranks=None, compute_core=True):
    """High Order Singular Value Decomposition.

    Args:
        tensor (Tensor): Array-like tensor to decomposition.
        ranks (list or int or None): Specify size of every mode component. Should have size equals to order of tensor.
        If given int, then extend to a list of the same value. If given none, use shape of tensor as ranks.
    """
    order = tensor.ndim
    shape = list(tensor.shape)
    
    if ranks is None:
        ranks = tensor.shape
    elif isinstance(ranks, int):
        ranks = [ranks for i in range(order)]
    elif len(ranks) != order:
        raise ValueError('N-ranks is not compatible with tensor.')

    factors = [None] * order
    for i in range(order):
        u, _, _ = T.svd(unfold(tensor, i), full_matrices=False)
        if u.shape[1] < ranks[i]:
            print(f'{i} mode rank {ranks[i]} larger then size of sigma, use {u.shape[0]} instead.')
            ranks[i] = u.shape[1]
        factors[i] = u[:, :ranks[i]]
        
    if compute_core:
        g = tensor
        for i in range(order):
            shape[i] = factors[i].shape[1]
            g = fold(factors[i].T @ unfold(g, i), i, shape)
            
        return g, factors 
    else:
        return factors


def reconstruct_hosvd(g: Tensor, factors: list) -> Tensor:
    order = g.ndim
    shape = list(g.shape)

    for i in range(order):
        shape[i] = factors[i].shape[0]
        g = fold(factors[i] @ unfold(g, i), i, shape)

    return g

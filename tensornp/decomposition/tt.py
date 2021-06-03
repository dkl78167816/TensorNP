'''
@Author: duke
@Date: 2021-06-03 15:43:21
@LastEditTime: 2021-06-03 21:05:55
@LastEditors: duke
@Description: some code
'''
from numpy.core.fromnumeric import cumprod, reshape
from .. import backend as T
from ..import Tensor


def tensor_train(tensor: Tensor, ranks):
    """Tensor-Train decomposition via recursive SVD.

    Args:
        tensor (Tensor): Tensor to decomposite.
        rank (int, int list): Maximum allowable TT rank of the factors. if int, then this is the same for all the factors. 
        if int list, then rank[k] is the rank of the kth factor.

    Returns:
        list : TT factors
    """
    # rank = validate_tt_rank(tl.shape(input_tensor), rank=rank)
    tensor_size = tensor.shape
    n_dim = len(tensor_size)

    c = tensor
    factors = [None] * n_dim

    for k in range(n_dim - 1):
        # Reshape the unfolding matrix of the remaining factors
        c = T.reshape(c, (ranks[k]*tensor_size[k], -1))

        # SVD of unfolding matrix
        (n_row, n_column) = c.shape
        ranks[k+1] = min(n_row, n_column, ranks[k+1])

        u, s, v = T.svd(c, full_matrices=False)
        u = u[:, :ranks[k+1]]
        s = s[:ranks[k+1]]
        v = v[:ranks[k+1], :]

        # Get kth TT factor
        factors[k] = T.reshape(u, (ranks[k], tensor_size[k], ranks[k+1]))

        # if(verbose is True):
        #     print("TT factor " + str(k) + " computed with shape " + str(factors[k].shape))

        # Get new unfolding matrix for the remaining factors
        c = T.reshape(s, (-1, 1)) * v
        # _temp = T.zeros(v.shape[0])
        # _temp[:len(s)] = s
        # c = _temp * v.T
        # print(c.shape)

    # Getting the last factor
    factors[-1] = T.reshape(c, (c.shape[0], c.shape[1], 1))

    return factors


def reconstruct_tt(factors):
    full_shape = [f.shape[1] for f in factors]
    full_tensor = T.reshape(factors[0], (full_shape[0], -1))

    for factor in factors[1:]:
        rank_prev, _, rank_next = factor.shape
        factor = T.reshape(factor, (rank_prev, -1))
        full_tensor = T.dot(full_tensor, factor)
        full_tensor = T.reshape(full_tensor, (-1, rank_next))

    return T.reshape(full_tensor, full_shape)

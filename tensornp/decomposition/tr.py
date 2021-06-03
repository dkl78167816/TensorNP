'''
@Author: duke
@Date: 2021-05-14 12:13:14
@LastEditTime: 2021-06-03 22:05:00
@LastEditors: duke
@Description: some code
'''
from functools import reduce

from numpy.core.fromnumeric import reshape
from .. import backend as T
from .. import Tensor
from ..linalg import unfold, fold, norm
import logging


def k_unfolding(tensor: Tensor, mode: int):
    sz_x = reduce(lambda a, b: a * b, tensor.shape[:mode + 1], 1)
    sz_y = reduce(lambda a, b: a * b, tensor.shape[mode + 1:], 1)
    T.reshape(tensor, (sz_x, sz_y))  


def tensor_ring(tensor: Tensor, ranks):
    shape = tensor.shape
    N = tensor.ndim
    
    cores = [None] * N

    # Roughly, capital letters correspond to matrices and the addition "_ten"
    # means it's a tensor, which is used when there's an under bar in the
    # algorithm in [Ah20].

    C = T.reshape(tensor, (shape[0], T.cumprod(shape[1:])))
    Z = C @ T.randn(T.cumprod(shape[1:]), ranks[N-1] * ranks[0])
    Q, _, _ = T.svd(Z, full_matrices=False)
    Q = Q[:, :ranks[N-1] * ranks[0]]
    cores[0] = T.transpose(T.reshape(Q, [shape[0], ranks[N-1], ranks[0]]), [1, 0, 2])
    
    C_ten = T.reshape(Q.T @ C, (Q.shape[1], *shape[1:]))
    C_ten = T.reshape(C_ten, [ranks[N-1], ranks[0], T.cumprod(shape[1:])])
    C_ten = T.transpose(C_ten, [1, 2, 0])
    C_ten = T.reshape(C_ten, [ranks[0] * shape[1], T.cumprod(shape[2:]), ranks[N-1]])

    for n in range(1, N-1):
        C = T.reshape(C_ten, [ranks[n-1]*shape[n], T.cumprod(shape[n+1:])*ranks[N-1]])
        Z = C @ T.randn(T.cumprod(shape[n+1:]) * ranks[N-1], ranks[n])
        Q, _, _ = T.svd(Z, full_matrices=False)
        Q = Q[:, :ranks[n]]
        cores[n] = T.reshape(Q, [ranks[n-1], shape[n], ranks[n]])
        C_ten = T.reshape(Q.T @ C, [ranks[n], T.cumprod(shape[n+1:]), ranks[N-1]])

    cores[N-1] = C_ten

    return cores


def tr_v2(tensor: Tensor, ranks):
    shape = tensor.shape
    ndim = tensor.ndim
    cores = [None] * ndim
    C = tensor
    
    r_old = 1
    n = shape[0]
    C = T.reshape(C, (r_old * ndim, -1))
    
    (u, s, v) = T.svd(C, full_matrices=False)
    r0 = ranks[-1]
    r_new = r0 * ranks[0]
    u = u[:, :r_new]
    _temp = T.zeros((r_new, r_new))
    _temp[:r_new, :r_new] = T.diag(s[:r_new])
    s = _temp
    v = v[:, :r_new]

    u_new = T.zeros((r_old * r0, ndim, r_new // r0))
    for i in range(r0):
        u_new[i, :, :] = u[:, (i-1)*r_new//r0+1:i*r_new//r0 + 1]
    cores[0] = u_new
    C = s @ v.T

    C_new = T.zeros((r_new//r0, C.size//r_new, r0))
    for i in range(r0):
        C_new[:, :, i] = C[(i-1)*(r_new//r0)+1:i*(r_new//r0) + 1, :]
    C = C_new
    C = T.reshape(C, (r_new//r0, shape[1], r0))
    # C = T.reshape(C, (r_new//r0, shape[1], shape[-1] * r0))
    # C = T.reshape(C, (r_new//r0, C.size//(r_new/r0)))
    r_old = r_new // r0

    for i in range(1, ndim-1):
        n = shape[i]
        C = T.reshape(C, (r_old*n, C.size//(r_old*n)))
        (u, s, v) = T.svd(C, full_matrices=False)
        r_new = ranks[i]
        u = u[:, :r_new]
        _temp = T.zeros((r_new, r_new))
        _temp[:r_new, :r_new] = T.diag(s[:r_new])
        s = _temp
        v = v[:, :r_new]
        cores[i] = T.reshape(u, (r_old, n, r_new))
        C = s @ v.T
        r_old = r_new
        
    C = reshape(C, (r_old, shape[-1], r0))
    cores[-1] = C

    return cores


def reconstruct_tr(factors):
    full_shape = [f.shape[1] for f in factors]
    full_tensor = T.reshape(factors[0], (full_shape[0], -1))

    for factor in factors[1:]:
        rank_prev, _, rank_next = factor.shape
        factor = T.reshape(factor, (rank_prev, -1))
        full_tensor = T.dot(full_tensor, factor)
        full_tensor = T.reshape(full_tensor, (-1, rank_next))

    return T.reshape(full_tensor, full_shape)
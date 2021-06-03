'''
@Author: duke
@Date: 2021-05-14 22:43:36
@LastEditTime: 2021-06-03 17:30:49
@LastEditors: duke
@Description: tensor singular value decomposition
'''
from .. import backend as T
from .. import Tensor
from ..linalg import tensor_transpose, t_product


def t_svd(tensor: Tensor):
    """Tensor SVD for 3 order tensor.

    Args:
        tensor (Tensor): 3 order tensor.

    Returns:
        U, S, V: 3 tensors. Result of t-svd.

    See https://github.com/andrewssobral/mtt

    """
    m = tensor.shape[0]
    n = tensor.shape[1]

    tensor = T.fft(tensor, axis=2)
    u = T.zeros((m, m, tensor.shape[2]))
    s = T.zeros((m, n, tensor.shape[2]))
    v = T.zeros((n, n, tensor.shape[2]))
    for i in range(tensor.shape[2]):
        _u, _s, _v = T.svd(tensor[:, :, i])
        u[:, :, i] = _u
        s[0:min(n, m), 0:min(n, m), i] = T.diag(_s)  
        v[:, :, i] = _v
    
    u = T.real(T.ifft(u, axis=2))
    s = T.real(T.ifft(s, axis=2))
    v = T.real(T.ifft(v, axis=2))

    return u, s, v


def reconstruct_t_svd(u, s, v):
    """reconstruct tensor in t_svd format.

    Returns:
        Tensor: a three order tensor.
    """
    # return t_product(t_product(u, s), tensor_transpose(v))

    return t_product(t_product(u, s), v)


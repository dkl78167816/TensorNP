'''
@Author: duke
@Date: 2021-05-04 22:11:54
@LastEditTime: 2021-06-03 17:30:32
@LastEditors: duke
@Description: Base operation
'''
from .core import seq_kr
from .. import backend as T
from .. import Tensor


def unfold(tensor, mode):
    """Returns the mode-`n` unfolding of `tensor` with modes starting at `0`.
    
    args:
        tensor : ndarray
        mode : int, default is 0,indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``
    
    return:
        ndarray: unfolded_tensor of shape ``(tensor.shape[mode], -1)``
    """
    return T.reshape(T.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


def fold(tensor, mode: int, shape):
    """Given a mode-`mode` unfolding tensor, fold to specific shape

    Args:
        tensor ([type]): [description]
        mode (int): [description]
        shape ([type]): [description]

    Returns:
        [type]: [description]
    """
    shape = list(shape)
    shape.insert(0, shape.pop(mode))
    return T.moveaxis(T.reshape(tensor, shape), 0, mode)


def norm(tensor, ord=2, axis=None):
    """Norm of `tensor`. 

    This function is able to return one of an given `ord` norm of tensor, depending on the value of the `ord`.

    Args:
        tensor ([type]): Input tensor.
        ord (int, optional): Order of the norm. Support `'inf'` or `int`. Defaults to 2.
        axis (int | None, optional): It specifies the axis of `tensor` along to compute the vector norms. 
        If `axis` is `None`, then compute norm of flatten `tensor`. Defaults to None.

    Returns:
        float | ndarray: Norm of the tensor or vectors.
    """
    if ord == 'inf':
        return T.abs(tensor).max(axis=axis)
    if ord == 1:
        return T.abs(tensor).sum(axis=axis)
    else:
        return T.power(T.abs(tensor), ord).sum(axis=axis) ** (1 / ord)


def mttkrp(tensor: Tensor, factors: Tensor, mode: int):
    return unfold(tensor, mode) @ seq_kr(factors, exclude=mode, reverse=False)


def tensor_transpose(tensor: Tensor):
    if tensor.ndim != 3:
        raise ValueError('tensor transpose needs 3 dimension tensors as parameters.')
    
    ret_tensor = T.zeros((tensor.shape[1], tensor.shape[0], tensor.shape[2]))
    
    ret_tensor[..., 0] = tensor[..., 0].T
    for i in range(1, tensor.shape[2]):
        ret_tensor[..., -i] = tensor[..., i].T

    return ret_tensor


def tensor_transpose_v2(tensor: Tensor):
    return tensor.transpose(1, 0, 2)


def conv_circ(signal, ker):
    '''Circular convolution
        
    Args:
        signal (Tensor): real 1D array
        ker (Tensor): real 1D array. signal and ker must have same shape
    '''
    return T.real(T.ifft(T.fft(signal) * T.fft(ker)))


def conv_circ_v2(signal, ker):
    # anather implementation of circular convolution.
    y = T.zeros(len(ker))
    _temp = T.zeros(len(ker))
    for i in range(len(ker)):
        for j in range(len(ker)):
            _temp[(j + i) % len(ker)] = signal[i] * ker[j] 
        print(_temp)
        y = y + _temp
    return y


def t_product(a: Tensor, b: Tensor):
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError('t-product needs 3 dimension tensors as parameters.')

    if a.shape[2] != b.shape[2] or a.shape[1] != b.shape[0]:
        raise ValueError('Shape of tensors are not compatible, please check tensors again.')

    ret_tensor = T.zeros((a.shape[0], b.shape[1], a.shape[2]))
    for i in range(ret_tensor.shape[0]):
        for j in range(ret_tensor.shape[1]):
            _temp = T.zeros(a.shape[2])
            for k in range(a.shape[1]):
                _temp = _temp + conv_circ(a[i, k, :], b[k, j, :])
                
            ret_tensor[i, j, :] = _temp

    return ret_tensor


def t_product_v2(a: Tensor, b: Tensor):
    # anather implementation of t_product
    def circ(tensor):
        n1 = tensor.shape[0]
        n2 = tensor.shape[1]
        n3 = tensor.shape[2]
        ret_matrix = T.zeros((n3 * n1, n3 * n2))
        
        for i in range(n3):
            for j in range(n3):
                ret_matrix[i*n1:(i+1)*n1, j*n2:(j+1)*n2] = tensor[:, :, (0 - j + i + n3) % n3]
        
        return ret_matrix
                
    def matvec(tensor):
        n1 = tensor.shape[0]
        n2 = tensor.shape[1]
        n3 = tensor.shape[2]
        ret_matrix = T.zeros((n3 * n1, n2))

        for i in range(n3):
            ret_matrix[i*n1:(i+1)*n1, :] = tensor[:, :, i]

        return ret_matrix

    return fold(circ(a) @ matvec(b), 2, (a.shape[0], b.shape[1], a.shape[2]))


def t_product_v3(a, b):
    def circmat(Au, s):
        n1, n2, n3 = s
        Auc = T.zeros((n1*n3, n2*n3))
        idx = 1
        shi = n1
        for k in range(n3):
            if k == 0:
                Auc[:, idx-1:idx+n2-1] = Au[:, :]
            else:
                Auc[:, idx-1:idx+n2-1] = T.roll(Au, shi)
                shi = shi + n1
            idx = idx + n2

        return Auc

    Au = unfold(a, 0).T
    Bu = unfold(b, 1)
    
    Auc = circmat(Au, a.shape)
    print(Auc.shape)
    print(Bu.shape)
    AucBu = Auc @ Bu.T
    
    return fold(AucBu, mode=2, shape=a.shape)
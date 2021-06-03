'''
@Author: duke
@Date: 2021-05-06 15:01:51
@LastEditTime: 2021-06-03 14:12:49
@LastEditors: duke
@Description: Numpy backend.
'''
import numpy as np


class NumpyBackend(object):
    backend_name = 'numpy'

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, dtype=None):
        return np.array(data, dtype=dtype)

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, np.ndarray)

    @staticmethod
    def to_numpy(tensor):
        return np.copy(tensor)

    @staticmethod
    def shape(tensor):
        return tensor.shape

    @staticmethod
    def ndim(tensor):
        return tensor.ndim

    @staticmethod
    def clip(tensor, a_min=None, a_max=None):
        return np.clip(tensor, a_min, a_max)

    @staticmethod    
    def dot(a, b):
        return a.dot(b)

    @staticmethod
    def sort(tensor, axis, descending=False):
        if descending:
            return np.flip(np.sort(tensor, axis=axis), axis = axis)
        else:
            return np.sort(tensor, axis=axis)


for name in ['int64', 'int32', 'float64', 'float32', 'complex128', 'complex64', 'real',
            'max', 'min', 'power', 'mean', 'sum', 'prod', 'sign', 'abs', 'sqrt', 'sin', 'cos', 'log2', 'tensordot',
            'any', 'where', 'argmin', 'argmax', 'all',
            'copy', 'transpose', 'reshape', 'moveaxis', 'flip', 'concatenate', 'stack', 'conj', 'vstack',
            'arange', 'ones', 'zeros', 'zeros_like', 'eye', 'kron', 'diag', 'einsum', 'roll']:
    setattr(NumpyBackend, name, getattr(np, name))


for name in ['solve', 'qr', 'svd', 'eigh', 'pinv']:
    setattr(NumpyBackend, name, getattr(np.linalg, name))

for name in ['rand', 'randn', 'randint', 'seed']:
    setattr(NumpyBackend, name, getattr(np.random, name))

for name in ['fft', 'ifft']:
    setattr(NumpyBackend, name, getattr(np.fft, name))
    
class NumpyTensor(np.ndarray):
    def __init__(self) -> None:
        super().__init__()

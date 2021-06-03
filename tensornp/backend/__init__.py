'''
@Author: duke
@Date: 2021-05-06 21:38:06
@LastEditTime: 2021-06-03 20:50:05
@LastEditors: duke
@Description: some code
'''
import numpy as np
import importlib
import os
import warnings
import functools


_DEFAULT_BACKEND = 'numpy'
_ALLOW_BACKENDS = ['numpy', 'pytorch']
_backend = None
_Tensor = None


def _init_backend():
    backend_name = os.environ.get('TENSORNP_BACKEND', _DEFAULT_BACKEND)
    if backend_name not in _ALLOW_BACKENDS:
        msg = ("TENSORNP_BACKEND should be one of {}, got {}. Defaulting to {}'").format(
               ', '.join(map(repr, _ALLOW_BACKENDS)),
               backend_name, _DEFAULT_BACKEND)
        warnings.warn(msg, UserWarning)
        backend_name = _DEFAULT_BACKEND

    global _backend
    global _Tensor
    # relatively import, __name__ of submodule will loss
    # _backend = importlib.import_module(name=(backend_name + '_backend'), package='backend')
    # absolutely import, __name__ regular. Error will don't run as main file
    # _backend = importlib.import_module(name= backend_name + '_backend')
    if backend_name == 'numpy':
        from .numpy_backend import NumpyBackend, NumpyTensor
        _backend = NumpyBackend
        _Tensor = NumpyTensor
    elif backend_name == 'pytorch':
        from .pytorch_backend import PytorchBackend, PytorchTensor
        _backend = PytorchBackend
        _Tensor = PytorchTensor
    # import numpy_backend  
    # import .numpy_backend  # wrong
    # from .numpy_backend import .  # wrong


_init_backend()
# for name in ['int64', 'int32', 'float64', 'float32', 'complex128', 'complex64', 
#              'reshape', 'moveaxis', 'any',
#              'where', 'copy', 'transpose', 'arange', 'ones', 'zeros', 'flip',
#              'zeros_like', 'eye', 'kron', 'concatenate', 'max', 'min',
#              'all', 'mean', 'sum', 'prod', 'sign', 'abs', 'sqrt', 'argmin',
#              'argmax', 'stack', 'conj', 'diag', 'einsum', 'log2', 'tensordot', 'sin', 'cos', 
#              'backend_name', 'context', 'sort',
#              'tensor', 'is_tensor', 'to_numpy', 'shape', 'ndim', 'clip', 'dot',
#              'solve', 'qr', 'svd', 'eigh']:
#     setattr(__current_module, name, getattr(_backend, name))
context = _backend.context

int64 = _backend.int64
int32 = _backend.int32
float64 = _backend.float64
float32 = _backend.float32
complex128 = _backend.complex128
complex64 = _backend.complex64
reshape = _backend.reshape
moveaxis = _backend.moveaxis
where = _backend.where
copy = _backend.copy
transpose = _backend.transpose
arange = _backend.arange
ones = _backend.ones
zeros = _backend.zeros
flip = _backend.flip
zeros_like = _backend.zeros_like
eye = _backend.eye
kron = _backend.kron
concatenate = _backend.concatenate
max = _backend.max
min = _backend.min
all = _backend.all
mean = _backend.mean
sum = _backend.sum
prod = _backend.prod
sign = _backend.sign
abs = _backend.abs
sqrt = _backend.sqrt
argmin = _backend.argmin
argmax = _backend.argmax
stack = _backend.stack
vstack = _backend.vstack
conj = _backend.conj
diag = _backend.diag
einsum = _backend.einsum
log2 = _backend.log2
tensordot = _backend.tensordot
sin = _backend.sin
cos = _backend.cos
tensor = _backend.tensor
is_tensor = _backend.is_tensor
to_numpy = _backend.to_numpy
shape = _backend.shape
ndim = _backend.ndim
clip = _backend.clip
dot = _backend.dot
solve = _backend.solve
qr = _backend.qr
svd = _backend.svd
eigh = _backend.eigh
power = _backend.power
pinv = _backend.pinv


def cumprod(l):
    ret = 1
    for i in l:
        ret *= i
    return ret

def return_tensor(func):
    @functools.wraps(func)  
    def wrapper(*args, **kw): 
        return tensor(func(*args, **kw))
    return wrapper


randn = return_tensor(_backend.randn)
randint = return_tensor(_backend.randint)
rand = return_tensor(_backend.rand)
seed = _backend.seed

fft = _backend.fft
ifft = _backend.ifft
real = _backend.real
roll = _backend.roll

from numpy import testing

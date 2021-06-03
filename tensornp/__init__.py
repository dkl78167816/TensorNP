__version__ = '1.0.0'


from .backend import  (context, tensor, is_tensor, shape, ndim, to_numpy, copy,
                      concatenate, reshape, transpose, moveaxis, arange, ones,
                      zeros, zeros_like, eye, where, clip, max, min, argmax,
                      argmin, all, mean, sum, prod, sign, abs, sqrt, dot,
                      kron, solve, qr, stack, log2, sin, cos, vstack, power, pinv,
                      rand, randn, randint, seed, fft, ifft, real, roll, cumprod)

from .backend import _Tensor as Tensor

from .decomposition import *


__all__ = [
    'tensor', 'cp', 'tucker', 'Tensor', 'hosvd', 't_svd', 'tt', 'tr'
]

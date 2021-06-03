'''
@Author: duke
@Date: 2021-05-07 22:24:17
@LastEditTime: 2021-05-09 11:35:01
@LastEditors: duke
@Description: some code
'''
from typing import Iterable, Union
from .. import backend as T
from .. import Tensor


def kron(a: Tensor, b: Tensor) -> Tensor:
    """Kronecker product of two tensors.

    For calculate details, see https://en.wikipedia.org/wiki/Kronecker_product
    Args:
        a (Tensor): matrix.
        b (Tensor): matrix.

    Returns:
        Tensor: result matrix.
    """
    s1, s2 = T.shape(a)
    s3, s4 = T.shape(b)
    a = T.reshape(a, (s1, 1, s2, 1))
    b = T.reshape(b, (1, s3, 1, s4))
    return T.reshape(a * b, (s1 * s3, s2 * s4))


def kr(a: Tensor, b: Tensor) -> Tensor:
    """Column-wise Khtri-Rao Product

    Version 2, one line code with vectorization and np.kron.
    Which is the Kronecker product of every column of A and B.
    See detail with: https://en.wikipedia.org/wiki/Khatri%E2%80%93Rao_product
    
    Args:
        a (Tensor): matrix.
        b (Tensor): matrix.

    Returns:
        Tensor: result matrix.
    """
    # Ensure same column numbers 
    if a.shape[1] != b.shape[1]:
        raise ValueError('All the tensors must have the same number of columns')
    c = T.vstack([T.kron(a[:, k], b[:, k]) for k in range(a.shape[1])]).T
    return c


def seq_kr(matrices: list, exclude: Union[int, list, None] = None, reverse=False) -> Tensor:
    """Do Khatri-Rao Product in a sequence of matrices.

    Args:
        matrices (list): Matrices to compute. [A1, A2, A3, ...]
        exclude (Union[int, list, None], optional): Index of matrix which to ignore in compute. 
        Starts from 0. Defaults to None.
        reverse (bool, optional): If True, the order of the matrices is reversed. Defaults to False.

    Returns:
        Tensor: Matrix as a result of computation.
    """  
    # Generate index except exclude matrix
    idx = list(range(len(matrices)))
    if isinstance(exclude, int):
        idx.pop(exclude)
    if isinstance(exclude, Iterable):
        for i in exclude:
            idx.remove(i)

    if reverse:
        idx = idx[::-1]

    res = matrices[idx[0]]
    for i in range(1, len(idx)):
        res = kr(res, matrices[idx[i]])
    return res


def seq_kron(matrices: list, exclude: Union[int, list, None] = None, reverse=False) -> Tensor:
    """Do Kronecker Product in a sequence of matrices.

    Args:
        matrices (list): Matrices to compute. [A1, A2, A3, ...]
        exclude (Union[int, list, None], optional): Index of matrix which to ignore in compute. Starts from 0. Defaults to None.
        reverse (bool, optional): If True, the order of the matrices is reversed. Defaults to False.

    Returns:
        Tensor: Matrix as a result of computation.
    """
    idx = list(range(len(matrices)))
    if isinstance(exclude, int):
        idx.pop(exclude)
    if isinstance(exclude, Iterable):
        for i in exclude:
            idx.remove(i)

    if reverse:
        idx = idx[::-1]
        
    res = matrices[idx[0]]
    for i in range(1, len(idx)):
        res = kron(res, matrices[idx[i]])
    return res
'''
@Author: duke
@Date: 2021-05-08 14:30:59
@LastEditTime: 2021-05-08 22:53:03
@LastEditors: duke
@Description: some code
'''
import numpy as np
import logging
import unittest

from .. import kron, seq_kron, seq_kr
from ... import backend as T


class TestKron(unittest.TestCase): 
    def setUp(self) -> None:
        logging.basicConfig(level=logging.INFO)
        return super().setUp()
        
    def test_kronecker(self):
        """Test for kronecker product"""
        # Mathematical test
        a = T.tensor([[1, 2, 3], [3, 2, 1]])
        b = T.tensor([[2, 1], [2, 3]])
        true_res = T.tensor([[2, 1, 4, 2, 6, 3],
                            [2, 3, 4, 6, 6, 9],
                            [6, 3, 4, 2, 2, 1],
                            [6, 9, 4, 6, 2, 3]])
        res = kron(a, b)
        np.testing.assert_array_equal(true_res, res)

        # Another test
        a = T.tensor([[1, 2], [3, 4]])
        b = T.tensor([[0, 5], [6, 7]])
        true_res = T.tensor([[0, 5, 0, 10],
                            [6, 7, 12, 14],
                            [0, 15, 0, 20],
                            [18, 21, 24, 28]])
        res = kron(a, b)
        np.testing.assert_array_equal(true_res, res)
        # Adding a third matrices
        c = T.tensor([[0, 1], [2, 0]])
        # res = kron(kron(c, a), b)
        res = seq_kron([c, a, b])
        assert (res.shape == (a.shape[0]*b.shape[0]*c.shape[0], a.shape[1]*b.shape[1]*c.shape[1]))
        np.testing.assert_array_equal(res[:4, :4], c[0, 0]*true_res)
        np.testing.assert_array_equal(res[:4, 4:], c[0, 1]*true_res)
        np.testing.assert_array_equal(res[4:, :4], c[1, 0]*true_res)
        np.testing.assert_array_equal(res[4:, 4:], c[1, 1]*true_res)

        # Test for the reverse argument
        temp_a = a.copy()
        temp_b = b.copy()
        res = kron(a, b)
        np.testing.assert_array_equal(res[:2, :2], a[0, 0]*b)
        np.testing.assert_array_equal(res[:2, 2:], a[0, 1]*b)
        np.testing.assert_array_equal(res[2:, :2], a[1, 0]*b)
        np.testing.assert_array_equal(res[2:, 2:], a[1, 1]*b)
        # Check that the original list has not been reversed
        np.testing.assert_array_equal(temp_a, a)
        np.testing.assert_array_equal(temp_b, b)

        # Check the returned shape
        shapes = [[2, 3], [4, 5], [6, 7]]
        W = [T.tensor(np.random.randn(*shape)) for shape in shapes]
        res = seq_kron(W)
        assert (res.shape == (48, 105))

        # Khatri-rao is a column-wise kronecker product
        shapes = [[2, 1], [4, 1], [6, 1]]
        W = [T.tensor(np.random.randn(*shape)) for shape in shapes]
        # res = kron(kron(W[0], W[1]), W[2])
        res = seq_kron(W)
        assert (res.shape == (48, 1))

        # Khatri-rao product is a column-wise kronecker product
        # kr = kr(W)
        kr_result = seq_kr(W)
        for i, shape in enumerate(shapes):
            np.testing.assert_array_almost_equal(res, kr_result)

        a = T.tensor([[1, 2],
                    [0, 3]])
        b = T.tensor([[0.5, 1],
                    [1, 2]])
        true_res = T.tensor([[0.5, 1., 1., 2.],
                            [1., 2., 2., 4.],
                            [0., 0., 1.5, 3.],
                            [0., 0., 3., 6.]])
        np.testing.assert_array_equal(kron(a, b),  true_res)
        reversed_res = T.tensor([[ 0.5,  1. ,  1. ,  2. ],
                                [ 0. ,  1.5,  0. ,  3. ],
                                [ 1. ,  2. ,  2. ,  4. ],
                                [ 0. ,  3. ,  0. ,  6. ]])
        np.testing.assert_array_equal(seq_kron([a, b], reverse=True),  reversed_res)

        # Test while skipping a matrix
        shapes = [[2, 3], [4, 5], [6, 7]]
        U = [T.tensor(np.random.randn(*shape)) for shape in shapes]
        res_1 = seq_kron(U, exclude=1)
        res_2 = seq_kron([U[0]] + U[2:])
        np.testing.assert_array_equal(res_1, res_2)

        res_1 = seq_kron(U, exclude=0)
        res_2 = seq_kron(U[1:])
        np.testing.assert_array_equal(res_1, res_2)

'''
@Author: duke
@Date: 2021-05-07 12:01:58
@LastEditTime: 2021-05-09 11:48:37
@LastEditors: duke
@Description: some code
'''
from functools import reduce
import unittest
import logging
from ...decomposition.cp import cp
from .. import norm
from ... import backend as T


class TestNorm(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.INFO)
        return super().setUp()

    def test_norm(self):
        v = T.tensor([1., 2., 3.])
        T.testing.assert_equal(norm(v,1), 6)

        A = T.reshape(T.arange(6), (3,2))
        T.testing.assert_equal(norm(A, 1), 15)

        column_norms1 = norm(A, 1, axis=0)
        row_norms1 = norm(A, 1, axis=1)
        T.testing.assert_array_equal(column_norms1, T.tensor([6., 9]))
        T.testing.assert_array_equal(row_norms1, T.tensor([1, 5, 9]))

        column_norms2 = norm(A, 2, axis=0)
        row_norms2 = norm(A, 2, axis=1)
        T.testing.assert_array_almost_equal(column_norms2, T.tensor([4.47213602, 5.91608]))
        T.testing.assert_array_almost_equal(row_norms2, T.tensor([1., 3.60555124, 6.40312433]))

        # limit as order->oo is the oo-norm
        column_norms10 = norm(A, 10, axis=0)
        row_norms10 = norm(A, 10, axis=1)
        T.testing.assert_array_almost_equal(column_norms10, T.tensor([4.00039053, 5.00301552]))
        T.testing.assert_array_almost_equal(row_norms10, T.tensor([1., 3.00516224, 5.05125666]))

        column_norms_oo = norm(A, 'inf', axis=0)
        row_norms_oo = norm(A, 'inf', axis=1)
        T.testing.assert_array_equal(column_norms_oo, T.tensor([4, 5]))
        T.testing.assert_array_equal(row_norms_oo, T.tensor([1, 3, 5]))

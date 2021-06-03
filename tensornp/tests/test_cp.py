'''
@Author: duke
@Date: 2021-05-06 22:46:55
@LastEditTime: 2021-05-27 16:15:37
@LastEditors: duke
@Description: some code
'''
import logging
from ..decomposition import cp
import unittest
from ..linalg.tenalg import reconstruct_tensor
from .. import backend as T
from sktensor import dtensor, cp_als


class TestCP(unittest.TestCase):
    """ Use scikit-tensor to check out decomposition correctness."""
    
    def setUp(self) -> None:
        logging.basicConfig(level=logging.INFO)
        return super().setUp()

    def test_cp_decomposition(self):
        test_data1 = T.array([[[0, 1, 3, 4], [4, 0, 2, 1], [4, 2, 3, 4]],
                             [[2, 4, 2, 3], [3, 3, 2, 4], [2, 3, 0, 2]]])
        test_data2 = T.array([[[3, 1, 1, 2], [1, 0, 3, 2], [3, 4, 0, 2]],
                              [[1, 2, 3, 3], [2, 3, 1, 0], [1, 2, 0, 2]]])
        factors, lamda = cp(test_data1, r=3, stop_iter=500, tol=1e-6, 
                            normalize_factor=True, random_seed=44)
        P, fit, itr = cp_als(dtensor(test_data1), 3, init='random')
        T.testing.assert_array_almost_equal(
            reconstruct_tensor(factors, lamda, (2, 3, 4)), 
            P.toarray(),
            decimal=0)

        factors, lamda = cp(test_data2, r=3, stop_iter=500, tol=1e-6, 
                            normalize_factor=True, random_seed=44)
        P, fit, itr = cp_als(dtensor(test_data2), 3, init='random')
        T.testing.assert_array_almost_equal(
            reconstruct_tensor(factors, lamda, (2, 3, 4)), 
            P.toarray(),
            decimal=0)
            
    def test_cp_reconstruction(self):
        data = T.array([[[3, 1, 1, 2], [1, 0, 3, 2], [3, 4, 0, 2]],
                         [[1, 2, 3, 3], [2, 3, 1, 0], [1, 2, 0, 2]]])
        tensor = dtensor(data)
        P, fit, itr = cp_als(tensor, 3, init='random')
        T.testing.assert_array_almost_equal(P.toarray(), reconstruct_tensor(P.U, P.lmbda, (2, 3, 4)))

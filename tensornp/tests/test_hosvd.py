'''
@Author: duke
@Date: 2021-05-06 22:46:55
@LastEditTime: 2021-05-31 11:06:30
@LastEditors: duke
@Description: some code
'''
import logging
from ..decomposition import hosvd 
import unittest
from ..decomposition.hosvd import reconstruct_tucker
from .. import backend as T
from sktensor import dtensor
from sktensor import tucker


class TestHOSVD(unittest.TestCase):
    
    def setUp(self) -> None:
        logging.basicConfig(level=logging.INFO)
        return super().setUp()

    def test_correctness(self):
        test_data1 = T.array([[[0, 1, 3, 4], [4, 0, 2, 1], [4, 2, 3, 4]],
                             [[2, 4, 2, 3], [3, 3, 2, 4], [2, 3, 0, 2]]])

        g, factors = hosvd(test_data1, [2, 2, 2])
        # logging.debug('g:', g)
        # logging.debug(print('factors:', factors))

        # sktensor cannot deal with test_data1
        test_data2 = T.random.randn(2, 3, 4)
        factors_sk, g_sk = tucker.hosvd(dtensor(test_data2), [2, 2, 2])
        g, factors = hosvd(test_data2, [2, 2, 2])

        T.testing.assert_array_almost_equal(T.abs(g), T.abs(g_sk))
        for i in range(len(factors)):
            T.testing.assert_array_almost_equal(T.abs(factors[i]), T.abs(factors_sk[i]))

    def test_reconstruct(self) -> None:
        test_data2 = T.array([[[3, 1, 1, 2], [1, 0, 3, 2], [3, 4, 0, 2]],
                              [[1, 2, 3, 3], [2, 3, 1, 0], [1, 2, 0, 2]]])
        g, factors = hosvd(test_data2, [2, 2, 2])
        res_data2 = reconstruct_tucker(g, factors)
        # print('reconstruct tensor 2: ', res_data2)

        test_data1 = T.array([[[0, 1, 3, 4], [4, 0, 2, 1], [4, 2, 3, 4]],
                             [[2, 4, 2, 3], [3, 3, 2, 4], [2, 3, 0, 2]]])
        g, factors = hosvd(test_data1, [2, 2, 2])
        res_data1 = reconstruct_tucker(g, factors)
        print(norm(res_data1 - test_data1))
        # print('reconstruct tensor 1: ', res_data1)


    # def test_speed_3ord(self):
    #     import time
    #     from sktensor import tucker 

    #     tensor_np = T.random.rand(2, 3, 4)
    #     tensor_sk = dtensor(tensor_np)

    #     print('Test speed for 3 order tensor')
    #     tick = time.time()
    #     tucker.hosvd(tensor_sk, [2, 3, 4])
    #     tock = time.time()
    #     print(f'sktensor: {tock - tick}')
    #     a, b = hosvd(tensor_np, ranks=[2, 3, 4])
    #     tick = time.time()
    #     print(f'tensornp: {tick - tock}')


    # def test_speed_10ord(self):
    #     import time
    #     from sktensor import tucker 

    #     tensor_np = T.random.rand(2, 3, 4, 5, 6, 7, 8, 9)
    #     tensor_sk = dtensor(tensor_np)
        
    #     print('Test speed for 8 order tensor')
    #     tick = time.time()
    #     tucker.hosvd(tensor_sk, [2, 3, 4, 5, 6, 7, 8, 9])
    #     tock = time.time()
    #     print(f'sktensor: {tock - tick}')
    #     hosvd(tensor_np, ranks=[2, 3, 4, 5, 6, 7, 8, 9])
    #     tick = time.time()
    #     print(f'tensornp: {tick - tock}')
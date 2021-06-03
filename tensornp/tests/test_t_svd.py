'''
@Author: duke
@Date: 2021-05-06 22:46:55
@LastEditTime: 2021-05-24 22:22:43
@LastEditors: duke
@Description: some code
'''
import logging
from ..decomposition import t_svd
import unittest
from .. import backend as T


class TestT_SVD(unittest.TestCase):
    """ Use scikit-tensor to check out decomposition correctness."""
    
    def setUp(self) -> None:
        logging.basicConfig(level=logging.INFO)
        return super().setUp()

    def test_speed_3ord(self):
        import time
        
        tensor_np = T.random.rand(2, 3, 4)
        
        print('Test speed for 3 order tensor')
        tick = time.time()
        t_svd(tensor_np)
        tock = time.time()
        print(f'tensornp: {tock - tick}')

    def test_speed_10ord(self):
        import time

        tensor_np = T.random.rand(2, 3, 4, 5, 6, 7, 8, 9)

        
        print('Test speed for 8 order tensor')
        tick = time.time()
        t_svd(tensor_np)
        tock = time.time()
        print(f'tensornp: {tock - tick}')
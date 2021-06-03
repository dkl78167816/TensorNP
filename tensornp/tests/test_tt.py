'''
@Author: duke
@Date: 2021-05-06 22:46:55
@LastEditTime: 2021-05-24 22:13:13
@LastEditors: duke
@Description: some code
'''
import logging
from ..decomposition import cp
import unittest
from ..linalg.tenalg import reconstruct_tensor
from .. import backend as T


class TestCP(unittest.TestCase):
    """ Use scikit-tensor to check out decomposition correctness."""
    
    def setUp(self) -> None:
        logging.basicConfig(level=logging.INFO)
        return super().setUp()

    # def test_speed_3ord(self):
    #     import time
    #     import tensorly as tl
    #     from tensorly.decomposition import tensor_train
        
    #     tensor_np = T.random.rand(2, 3, 4)
    #     tensor_ly = tl.tensor(tensor_np)

    #     print('Test speed for 3 order tensor')
    #     tick = time.time()
    #     tensor_train(tensor_ly, rank=2)
    #     tock = time.time()
    #     print(f'tensorly: {tock - tick}')


    # def test_speed_10ord(self):
    #     import time
    #     import tensorly as tl
    #     from tensorly.decomposition import tensor_train
        
    #     tensor_np = T.random.rand(2, 3, 4, 5, 6, 7, 8, 9)
    #     tensor_ly = tl.tensor(tensor_np)
        
    #     print('Test speed for 8 order tensor')
    #     tick = time.time()
    #     tensor_train(tensor_ly, rank=8)
    #     tock = time.time()
    #     print(f'tensorly: {tock - tick}')

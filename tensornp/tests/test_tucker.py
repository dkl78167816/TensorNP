'''
@Author: duke
@Date: 2021-05-06 22:46:55
@LastEditTime: 2021-05-24 22:00:05
@LastEditors: duke
@Description: some code
'''
import logging
from ..decomposition import cp
import unittest
from ..linalg.tenalg import reconstruct_tensor
from .. import backend as T
from sktensor import dtensor, cp_als


class TestTucker(unittest.TestCase):
    """ Use scikit-tensor to check out decomposition correctness."""
    
    def setUp(self) -> None:
        logging.basicConfig(level=logging.INFO)
        return super().setUp()

    # def test_speed_3ord(self):
    #     import time
    #     import sktensor
    #     from tensorly.decomposition import tucker
    #     import tensorly as tl
        
    #     tensor_np = T.random.rand(2, 3, 4)
    #     tensor_sk = dtensor(tensor_np)
    #     tensor_ly = tl.tensor(tensor_np)
        
    #     print('Test speed for 3 order tensor')
    #     tick = time.time()
    #     sktensor.tucker.hooi(tensor_sk, [2, 3, 4], init='nvecs')
    #     tock = time.time()
    #     print(f'sktensor: {tock - tick}')
    #     tucker(tensor_ly, rank=[2, 3, 4], init='svd')
    #     tick = time.time()
    #     print(f'tensorly: {tick - tock}')

    # def test_speed_10ord(self):
    #     import time
    #     import sktensor
    #     from tensorly.decomposition import tucker
    #     import tensorly as tl
        
    #     tensor_np = T.random.rand(2, 3, 4, 5, 6, 7, 8, 9)
    #     tensor_sk = dtensor(tensor_np)
    #     tensor_ly = tl.tensor(tensor_np)
        

    #     print('Test speed for 8 order tensor')

    #     tick = time.time()
    #     sktensor.tucker.hooi(tensor_sk, [2, 3, 4, 5, 6, 7, 8, 9], init='nvecs')
    #     tock = time.time()
    #     print(f'sktensor: {tock - tick}')
    #     tucker(tensor_ly, rank=[2, 3, 4, 5, 6, 7, 8, 9], init='svd')
    #     tick = time.time()
    #     print(f'tensorly: {tick - tock}')

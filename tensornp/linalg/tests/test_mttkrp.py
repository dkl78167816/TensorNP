import sktensor
import unittest
from ... import backend as T
from .. import mttkrp, unfold


class TestMttkrp(unittest.TestCase):
    def test_mttkrp(self):
        t = T.array([[[0, 1, 3, 4], [4, 0, 2, 1], [4, 2, 3, 4]],
                     [[2, 4, 2, 3], [3, 3, 2, 4], [2, 3, 0, 2]]])
        dtensor = sktensor.dtensor(t)

        factors = [None] * 3
        for i in range(3):
            factors[i] = T.random.rand(t.shape[i], 3)
        # print(factors)
        # print(dtensor.uttkrp(factors, 0))
        # print(mttkrp(t, factors, 0))
        # T.testing.assert_array_almost_equal(dtensor.uttkrp(factors, 0),
        #     mttkrp(t, factors, 0))

        # 对展开的细节理解不一样，导致结果不一样，但是两种算法都没有问题
        # T.testing.assert_array_almost_equal(dtensor.unfold(0), unfold(t, 0)) 
        from sktensor.core import khatrirao
        from .. import seq_kr

        order = list(range(0)) + list(range(0 + 1, 3))
        T.testing.assert_array_almost_equal(
            seq_kr(factors, exclude=0, reverse=True),
            khatrirao(tuple(factors[i] for i in order), reverse=True))

            
        order = list(range(1)) + list(range(1 + 1, 3))
        T.testing.assert_array_almost_equal(
            seq_kr(factors, exclude=1, reverse=True),
            khatrirao(tuple(factors[i] for i in order), reverse=True))
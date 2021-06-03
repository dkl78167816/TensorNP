'''
@Author: duke
@Date: 2021-05-09 11:39:04
@LastEditTime: 2021-05-11 10:14:03
@LastEditors: duke
@Description: some code
'''
import numpy as np
import logging
import unittest

from .. import fold, unfold
from ... import backend as T


class TestFold(unittest.TestCase):
    def test_fold(self):
        """Test for fold
        """
        X = T.reshape(T.arange(24), (3, 4, 2))
        unfoldings = [T.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                [8, 9, 10, 11, 12, 13, 14, 15],
                                [16, 17, 18, 19, 20, 21, 22, 23]]),
                T.tensor([[0, 1, 8, 9, 16, 17],
                                [2, 3, 10, 11, 18, 19],
                                [4, 5, 12, 13, 20, 21],
                                [6, 7, 14, 15, 22, 23]]),
                T.tensor([[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                                [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]])]
        # hard coded example
        for mode in range(T.ndim(X)):
            T.testing.assert_array_equal(fold(unfoldings[mode], mode, X.shape), X)

        # check dims
        for i in range(T.ndim(X)):
            T.testing.assert_array_equal(X, fold(unfold(X, i), i, X.shape))

        # chain unfolding and folding
        X = T.tensor(np.random.random(2 * 3 * 4 * 5).reshape(2, 3, 4, 5))
        for i in range(T.ndim(X)):
            T.testing.assert_array_equal(X, fold(unfold(X, i), i, X.shape))

    def test_unfold(self):
        """Test for unfold
        1. We do an exact test.
        2. Second,  a test inspired by the example in Kolda's paper:
            Even though we use a different definition of the unfolding,
            it should only differ by the ordering of the columns
        """
        X = T.reshape(T.arange(24), (3, 4, 2))
        unfoldings = [T.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                [8, 9, 10, 11, 12, 13, 14, 15],
                                [16, 17, 18, 19, 20, 21, 22, 23]]),
                        T.tensor([[0, 1, 8, 9, 16, 17],
                                [2, 3, 10, 11, 18, 19],
                                [4, 5, 12, 13, 20, 21],
                                [6, 7, 14, 15, 22, 23]]),
                        T.tensor([[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                                [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]])]
        for mode in range(T.ndim(X)):
            unfolding = unfold(X, mode=mode)
            T.testing.assert_array_equal(unfolding, unfoldings[mode])
            T.testing.assert_array_equal(T.reshape(unfolding, (-1, )),
                                T.reshape(unfoldings[mode], (-1,)))
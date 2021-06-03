'''
@Author: duke
@Date: 2021-05-29 09:36:18
@LastEditTime: 2021-05-29 09:40:49
@LastEditors: duke
@Description: some code
'''
from typing import Iterable


class CPTensor():
    def __init__(self, factors, lamda: float) -> None:
        if not isinstance(factors, Iterable):
            raise ValueError('Factors are not list.')
        self.factors = list(factors)
        self.lamda = lamda

    def to_tensor(self):
        pass

    
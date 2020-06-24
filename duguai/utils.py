# -*- coding: utf-8 -*-
from typing import Union, List

import numpy as np


def is_in(short_arr: Union[List, np.ndarray], long_arr: Union[List, np.ndarray]) -> bool:
    """
    判断short_arr是否在long_arr中。

    Examples
    >>> is_in([1, 2, 3], [1, 2, 3, 4, 5, 6])
    True
    @param short_arr: 较短的数组
    @param long_arr: 较长的数组
    @return:
    """
    return sum(np.array(np.in1d(short_arr, long_arr))) == len(short_arr)

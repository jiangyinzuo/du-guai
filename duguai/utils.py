# -*- coding: utf-8 -*-
import numpy as np


def is_in(arr1: np.ndarray, arr2: np.ndarray):
    return sum(np.in1d(arr1, arr2)) == len(arr1)

# -*- coding: utf-8 -*-

from random import sample
from typing import List

import numpy as np


def get_action(state: np.ndarray, actions: List[int]) -> int:
    """
    通过Q-learning算法获取下一步动作
    @param state:
    @param actions:
    """
    return sample(actions, 1)[0]

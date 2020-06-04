# -*- coding: utf-8 -*-
"""
叫地主算法
"""
import numpy as np

from duguai.card_def import *


def has_g(raw_data: np.ndarray) -> int:
    """
    大小王情况
    @param raw_data: 长度为17的一维数组，代表手牌
    @return: 0: 没有大小王，1: 只有小王，2: 只有大王，3: 大小王都有
    """
    result = 0
    if raw_data[-1] == 15:
        result += 2
    if raw_data[-1] == 14 or raw_data[-2] == 14:
        result += 1
    return result


def bomb_count(raw_data: np.ndarray) -> int:
    """
    除大小王以外的炸弹数量
    @param raw_data: 长度为17的一维数组，代表手牌
    @return: 炸弹数
    """
    return sum(raw_data[i] == raw_data[i - 3] for i in range(3, 17))


def card2_count(raw_data: np.ndarray) -> int:
    """
    2的数量
    @param raw_data: 长度为17的一维数组，代表手牌
    @return: 2的数
    """
    return sum(raw_data == CARD_2)


def process(raw_data: np.ndarray) -> np.ndarray:
    """
    预处理原始手牌，转换成特征向量
    @param raw_data:
    @return:
    """
    return np.array([has_g(raw_data), bomb_count(raw_data), card2_count(raw_data)])


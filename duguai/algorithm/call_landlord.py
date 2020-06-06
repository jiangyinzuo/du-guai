# -*- coding: utf-8 -*-
"""
叫地主算法
"""
import numpy as np
from sklearn.svm import LinearSVC

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


def get_svc() -> LinearSVC:
    """
    获取训练出来的支持向量机分类器
    @return: LinearSVC
    """
    svc = LinearSVC(5.24e-05)
    svc.n_iter_ = 7
    svc.coef_ = np.array([[0.19581239, 0.03330529, 0.10988893]])
    svc.classes_ = np.array([0, 1])
    svc.intercept_ = np.array([-0.06151605])
    return svc


class CardZscore:
    def __init__(self, mean, scale):
        self.mean = mean
        self.scale = scale

    def zscore(self, X):
        return (X - self.mean) / self.scale


def get_card_zscore():
    return CardZscore(np.array([0.95726285, 0.09313241, 1.24184783]),
                      np.array([1.02556338, 0.29399838, 0.90237814]))

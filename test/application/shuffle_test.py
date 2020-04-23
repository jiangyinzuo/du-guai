# -*- coding: utf-8 -*-
import numpy

from duguai.application.shuffle import shuffle


def test_shuffle():
    """
    测试洗牌函数
    """
    cards: numpy.ndarray = shuffle()
    assert type(cards) == numpy.ndarray
    assert len(cards) == 54

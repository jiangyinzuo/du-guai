# -*- coding: utf-8 -*-
import numpy as np

from card.card_helper import card_split


def test_split():
    test_pair = [
        ([1, 1], [[1, 1]]),
        ([2, 2, 3, 4, 6], [[2, 2, 3, 4], [6]])
    ]
    for i in test_pair:
        res = card_split(np.array(i[0]))
        r = [np.array(j) for j in i[1]]
        for j in zip(res, r):
            assert (j[0] == j[1]).all()

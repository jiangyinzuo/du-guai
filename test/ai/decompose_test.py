# -*- coding: utf-8 -*-
from duguai.ai.decompose import *


def test_decomposed_value():
    test_data = [
        ([5, 5, 6, 6, 6], 5),
        ([5, 5, 6, 6, 7, 7], 6),
    ]
    for i in test_data:
        cards, d_value = i[0], i[1]
        assert Decomposer.decompose_value(cards) == d_value


def test_get_good_actions():
    decomposer = Decomposer()
    print(decomposer.get_good_play([5, 5, 6, 6, 7, 7, 7, 9, 9, 14, 15]))
    print(decomposer.get_good_play([3, 4, 5, 5, 6, 6, 7, 7, 7, 8, 9, 10, 11]))
    print(decomposer.get_good_play([5, 6, 7, 8, 9, 10, 10]))

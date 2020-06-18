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
    decomposer = PlayDecomposer()
    print(decomposer.get_good_plays([1, 1, 1, 1, 5, 5, 6, 6, 7, 7, 7, 9, 9, 10, 10, 11, 11, 14, 15]))
    print(decomposer.get_good_plays([3, 4, 5, 5, 6, 6, 7, 7, 7, 8, 9, 10, 11]))
    print(decomposer.get_good_plays([5, 6, 7, 8, 9, 10, 10]))


def test_good_solo():
    combo = Combo()
    combo.cards = [3]
    decomposer = FollowDecomposer()
    print(decomposer.get_good_follows([4, 5, 6, 6, CARD_2, CARD_2, CARD_G0, CARD_G1], combo))

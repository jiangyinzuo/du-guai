# -*- coding: utf-8 -*-
from duguai.ai.decompose import *


def test_decomposed_value():
    test_data = [
        ([5, 5, 6, 6, 6], 5),
        ([5, 5, 6, 6, 7, 7], 6),
    ]
    for i in test_data:
        cards, d_value = i[0], i[1]
        assert AbstractDecomposer.decompose_value(cards) == d_value


def test_get_good_actions():
    decomposer = PlayDecomposer()
    combo = Combo()
    combo.cards_view = '3 3 4 5 5 6 7 7 7 8 9 10 J Q K K A 2 g G'
    print(decomposer.get_good_plays(combo.cards))
    print(decomposer.get_good_plays([5, 6, 7, 7, 8, 8, 9, 10]))
    print(decomposer.get_good_plays([1, 1, 1, 1, 5, 5, 6, 6, 7, 7, 7, 9, 9, 10, 10, 11, 11, 14, 15]))
    print(decomposer.get_good_plays([3, 4, 5, 5, 6, 6, 7, 7, 7, 8, 9, 10, 11]))


def test_good_single():
    combo = Combo()
    combo.cards = [3]
    decomposer = FollowDecomposer()
    print(decomposer.get_good_follows([4, 5, 6, 6, CARD_2, CARD_2, CARD_G0, CARD_G1], combo))

    combo.cards = [3, 4, 5, 6, 7]
    print(decomposer.get_good_follows([5, 6, 7, 8, 9, 10, 11, 12], combo))
    combo.cards = [8]
    assert len(decomposer.get_good_follows([5, 7, CARD_G0, CARD_G1], combo)) == 2


def test_trio_with_one():
    combo = Combo()
    combo.cards = [3, 7, 7, 7]
    decomposer = FollowDecomposer()
    print(decomposer.get_good_follows([8, 8, 8, 9, 10], combo))
    print(decomposer.get_all_follows_no_carry([8, 8, 8, 9, 10], combo))


def test_shuttle():
    decomposer = PlayDecomposer()
    combo = Combo()
    combo.cards_view = '3 3 3 3 4 4 4 4 8'
    print(decomposer.get_good_plays(combo.cards))

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


def test_get_good_plays():
    decomposer = PlayDecomposer()
    combo = Combo()
    combo.cards_view = '3 3 4 5 5 6 7 7 7 8 9 10 J Q K K A 2 g G'
    print(decomposer.get_good_plays(combo.cards))

    test_data = [
        [5, 6, 7, 7, 8, 8, 9, 10],
        [1, 1, 1, 1, 5, 5, 6, 6, 7, 7, 7, 9, 9, 10, 10, 11, 11, 14, 15],
        [3, 4, 5, 5, 6, 6, 7, 7, 7, 8, 9, 10, 11]
    ]

    for i in test_data:
        cards = np.array(i)
        print(decomposer.get_good_plays(cards))


def test_takes():
    play_decomposer = PlayDecomposer()
    follow_decomposer = FollowDecomposer()
    test_cards = [
        ([1, 1, 4, 4, 4, 5, 5, 5, 6], 3, [6, 6, 3, 3, 3, 4, 4, 4], 2, 2)
    ]

    combo = Combo()
    for i in test_cards:
        combo.cards = i[2]
        cards = np.array(i[0])
        assert len(play_decomposer.get_good_plays(cards)) == i[1]
        print(follow_decomposer.get_good_follows(cards, combo))


def test_good_single():
    combo = Combo()
    combo.cards = [3]
    decomposer = FollowDecomposer()
    print(decomposer.get_good_follows(np.array([4, 5, 6, 6, CARD_2, CARD_2, CARD_G0, CARD_G1]), combo))

    combo.cards = [3, 4, 5, 6, 7]
    print(decomposer.get_good_follows(np.array([5, 6, 7, 8, 9, 10, 11, 12]), combo))
    combo.cards = [8]
    print(decomposer.get_good_follows(np.array([5, 7, CARD_G0, CARD_G1]), combo))


def test_trio_with_one():
    combo = Combo()
    combo.cards = [3, 7, 7, 7]
    decomposer = FollowDecomposer()
    print(decomposer.get_good_follows(np.array([8, 8, 8, 9, 10]), combo))
    print(decomposer.get_good_follows(np.array([8, 8, 9, 10]), combo))
    print(decomposer.get_good_follows(np.array([8, 8, 8, 8, 9, 10]), combo))


def test_bomb():
    decomposer = PlayDecomposer()
    combo = Combo()
    combo.cards_view = '3 3 3 3 4 4 4 4 8'
    print(decomposer.get_good_plays(combo.cards))


def test_seq():
    decomposer = FollowDecomposer()
    combo = Combo()

    combo.cards = [4, 5, 6, 7, 8]
    print(decomposer.get_good_follows(np.array([5, 5, 6, 7, 8, 9]), combo))

# -*- coding: utf-8 -*-
import numpy as np

from duguai.ai.decompose import FollowDecomposer, PlayDecomposer
from duguai.card import CARD_2, CARD_G0, CARD_G1
from duguai.card.combo import Combo


def test_card2():
    decomposer = FollowDecomposer()
    combo = Combo()
    combo.cards_view = '5 6 7 8 9'

    print(decomposer.get_good_follows(np.array([CARD_2]), combo))
    combo.cards_view = '8 8 8 9 9'
    print(decomposer.get_good_follows(np.array([CARD_2]), combo))
    print(decomposer.get_good_follows(np.array([CARD_2, CARD_2]), combo))
    print(decomposer.get_good_follows(np.array([CARD_2, CARD_2, CARD_2]), combo))


def test_get_good_plays():
    decomposer = PlayDecomposer()
    combo = Combo()
    combo.cards_view = '3 3 4 5 5 6 7 7 7 8 9 10 J Q K K A 2 g G'
    print(decomposer.get_good_plays(combo.cards))

    test_data = [
        [5, 6, 7, 7, 8, 8, 9, 10],
        [1, 1, 1, 1, 5, 5, 6, 6, 7, 7, 7, 9, 9, 10, 10, 11, 11, 14, 15],
        [3, 4, 5, 5, 6, 6, 7, 7, 7, 8, 9, 10, 11],
        [1, 2, 3, 3, 5, 6, 7, 7, 7, 8, 8, 9, 10, 11, 11, 11, 12,
         13, 14, 15]
    ]

    for i in test_data:
        cards = np.array(i)
        print(decomposer.get_good_plays(cards))


def test_takes():
    play_decomposer = PlayDecomposer()
    follow_decomposer = FollowDecomposer()
    test_cards = [
        ([1, 1, 4, 4, 4, 5, 5, 5, 6], [6, 6, 3, 3, 3, 4, 4, 4])
    ]

    print(play_decomposer.get_good_plays(np.array([2, 5, 6, 7, 8, 9, 9, 10, 10, 10, 11, 12])))

    combo = Combo()
    for i in test_cards:
        combo.cards = i[1]
        cards = np.array(i[0])
        print(play_decomposer.get_good_plays(cards))
        print(follow_decomposer.get_good_follows(cards, combo))

    print(play_decomposer.get_good_plays(np.array([5, 7, 8, 9, 10, 10, 10, 11, 11, 11, 12, 12])))
    print(play_decomposer.get_good_plays(np.array([3, 3, 5, 7, 8, 9, 10, 10, 10, 11, 11, 11, 12, 12])))


def test_good_single():
    combo = Combo()
    combo.cards = [3]
    decomposer = FollowDecomposer()
    print(decomposer.get_good_follows(np.array([4, 5, 6, 6, CARD_2, CARD_2, CARD_G0, CARD_G1]), combo))

    combo.cards = [3, 4, 5, 6, 7]
    print(decomposer.get_good_follows(np.array([5, 6, 7, 8, 9, 10, 11, 12]), combo))
    combo.cards = [8]
    print(decomposer.get_good_follows(np.array([5, 7, CARD_G0, CARD_G1]), combo))

    combo.cards_view = '2'
    print(decomposer.get_good_follows(np.array([CARD_2]), combo))

    combo.cards_view = 'A A'
    print(decomposer.get_good_follows(np.array([CARD_2, CARD_2, CARD_2]), combo))


def test_trio_with_one():
    combo = Combo()
    combo.cards = [3, 7, 7, 7]
    decomposer = FollowDecomposer()
    print(decomposer.get_good_follows(np.array([8, 8, 8, 9, 10]), combo))
    print(decomposer.get_good_follows(np.array([8, 8, 9, 10]), combo))
    print(decomposer.get_good_follows(np.array([8, 8, 8, 8, 9, 10]), combo))

    combo.cards = [3, 3, 7, 7, 7]
    print(decomposer.get_good_follows(np.array([8, 8, 8, CARD_2, CARD_2]), combo))
    print(decomposer.get_good_follows(np.array([8, 8, 8, CARD_2]), combo))
    print(decomposer.get_good_follows(np.array([1, 1, 2, 3, 4, 5, 6, 9, 9, 9, 10, 10, 10, 11, 11, 11]), combo))


def test_trio():
    decomposer = PlayDecomposer()
    print(decomposer.get_good_plays(np.array([4, 4, 4, 5, 6, 6, 7, 7])))
    print(decomposer.get_good_plays(np.array([1, 1, 2, 3, 4, 5, 6, 9, 9, 9, 10, 10, 10, 11, 11, 11])))


def test_long_seq():
    combo = Combo()
    decomposer = FollowDecomposer()
    combo.cards_view = 'Q'
    print(decomposer.get_good_follows(np.array([1, 1, 2, 3, 4, 5, 5, 6, 7, 7, 8, 8, 8, 9, 10, 11, 12]), combo))


def test_four_takes():
    decomposer = PlayDecomposer()
    follow_decomposer = FollowDecomposer()
    combo = Combo()
    combo.cards_view = '4 4 4 4 7 8'
    play_hand = decomposer.get_good_plays(combo.cards)
    print(play_hand)
    print(follow_decomposer.get_good_follows(np.array([9, 9, 9, 9, 8, 10]), combo))


def test_seq():
    decomposer = FollowDecomposer()
    combo = Combo()

    combo.cards = [4, 5, 6, 7, 8]
    print(decomposer.get_good_follows(np.array([5, 5, 6, 7, 8, 9]), combo))
    print(decomposer.get_good_follows(np.array([CARD_G1, CARD_G0]), combo))

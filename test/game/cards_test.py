# -*- coding: utf-8 -*-

from card.cards import *


def test_cards():
    c1 = Combo()
    c1.cards_view = 'Q Q J'
    assert np.sum(c1.cards == [CARD_J, CARD_Q, CARD_Q]) == 3
    assert c1.bit_info == INVALID_BIT

    c1.cards = np.array([1, 1, 1, 4])
    assert c1.bit_info == 11301


def test_cmp_cards():
    c1 = Combo()
    c2 = Combo()

    c1.cards_view = '3 3 5 3 5'
    c2.cards_view = 'A A A J J'
    print(Combo.bit_is_valid(c2.bit_info, c1.bit_info))
    print(c2)
    assert c2 > c1

    test_di = {
        '5 7 K K K A A A': 12300 + CARD_A,
        '8  K 8 K K A A A': 12300 + CARD_A,
        '7 8 8 8': 11300 + CARD_8,
        '7 7 6 6 6': 21300 + CARD_6
               }

    for k, v in test_di.items():
        c1.cards_view = k
        assert c1.bit_info == v

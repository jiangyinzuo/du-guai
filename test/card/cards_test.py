# -*- coding: utf-8 -*-
from card import CARD_A, CARD_G0, CARD_2
from card.combo import Combo


def test_combo_type():
    combo = Combo()
    combo.cards = [CARD_A, CARD_A, CARD_A, CARD_2, CARD_G0]
    assert combo.is_valid()

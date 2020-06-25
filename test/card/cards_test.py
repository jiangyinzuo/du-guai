# -*- coding: utf-8 -*-
from duguai.card import CARD_A, CARD_G0, CARD_2
from duguai.card.combo import Combo


def test_combo_type():
    combo = Combo()
    combo.cards = [CARD_A, CARD_A, CARD_A, CARD_2, CARD_G0]
    assert not combo.is_valid()

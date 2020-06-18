# -*- coding: utf-8 -*-
from card.combo import Combo


def test_combo_type():
    combo = Combo()
    combo.cards_view = '3'
    assert combo.is_solo()

# -*- coding: utf-8 -*-
import numpy as np

from application.view import get_cards_view


def test_get_card_view():
    result: str = get_cards_view(np.array(range(1, 16)))
    assert result == '3 4 5 6 7 8 9 10 J Q K A 2 g G '

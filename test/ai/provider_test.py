# -*- coding: utf-8 -*-
import numpy as np

from ai.provider import Provider
from card.combo import Combo


def test_state_provider():

    test_data = [
        ([[1], [2], [3]], 0, 0),
        ([np.array([3]), np.array([10])], 1, 2),
        ([[15]], 3, 3),
        ([[1], [2], [13], [15]], 0, 3)
    ]

    # 测试 _f1_min, _f1_max
    for i in test_data:
        actions, min_value, max_value = i
        assert Provider.StateProvider._f_min(actions) == min_value
        assert Provider.StateProvider._f_max(actions) == max_value


def test_provide():
    provider = Provider(1)
    provider.add_landlord(1)
    result = provider.provide([3, 4, 5, 6, 7, 9], 6, 6, 1)
    print(result)

    combo = Combo()
    combo.cards = [13, 13]
    result = provider.provide([4, 4, 4, 4, 7], 1, 1, 2, combo)
    print(result)

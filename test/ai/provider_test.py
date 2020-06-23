# -*- coding: utf-8 -*-
import numpy as np

from ai.provider import PlayProvider


def test_provide():
    play_provider = PlayProvider(1)
    play_provider.add_landlord_id(1)
    result = play_provider.provide(np.array([3, 4, 5, 6, 7, 9]), 6, 6)
    print(result)

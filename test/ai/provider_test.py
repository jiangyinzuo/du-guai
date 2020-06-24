# -*- coding: utf-8 -*-
import numpy as np

from ai.executor import execute_play
from ai.provider import PlayProvider, FollowProvider
from card.combo import Combo


def test_provide():
    play_provider = PlayProvider(1)
    play_provider.add_landlord_id(1)
    result = play_provider.provide(np.array([3, 4, 5, 6, 7, 9]), 6, 6)
    print(result)

    result = play_provider.provide(
        np.array([[1, 1, 1, 1, 2, 2, 2, 4, 6, 6, 7, 8, 9, 9, 11, 12, 13, 13, 13, 13]]), 10, 10)
    print(result[0])
    print(execute_play(result[0], 24))


def test_follow():
    follow_provider = FollowProvider(1)
    follow_provider.add_landlord_id(1)
    combo = Combo()
    combo.cards = [3, 3]
    result = follow_provider.provide(2, 10, 10, np.array([7, 7, 12, 12, 12]), combo)
    print(result)
    combo.cards = [9, 9]
    result = follow_provider.provide(2, 10, 10, np.array([7, 7, 12, 12, 12]), combo)
    print(result)
    combo.cards = [4]
    result = follow_provider.provide(2, 10, 10, np.array([7, 8, 12, 12, 12]), combo)
    print(result)
    combo.cards = [9]
    result = follow_provider.provide(2, 10, 10, np.array([7, 8, 12, 12, 12]), combo)
    print(result)
    combo.cards = [13]
    result = follow_provider.provide(2, 10, 10, np.array([7, 8, 12, 12, 12]), combo)
    print(result)

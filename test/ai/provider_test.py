# -*- coding: utf-8 -*-
import numpy as np

from duguai.ai.provider import FollowProvider
from duguai.card.combo import Combo


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

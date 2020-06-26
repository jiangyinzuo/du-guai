# -*- coding: utf-8 -*-
from random import sample

import numpy as np

from ai.executor import execute_play
from duguai.ai.provider import FollowProvider, PlayProvider
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


def test_sorted_follow():
    follow_provider = FollowProvider(1)
    follow_provider.add_landlord_id(1)
    combo = Combo()
    combo.cards_view = '3 4 4 6 6 7 8 8 8 10 Q Q K K A 2 2'
    combo1 = Combo()
    combo1.cards_view = '3 3'
    result = follow_provider.provide(0, 18, 17, combo.cards, combo1)
    print(result)

    combo.cards_view = '3 3 7 7 7'
    combo1.cards_view = '2 2 8 8 8 4 4'
    result = follow_provider.provide(0, 18, 17, combo1.cards, combo)
    print(result)


def test_sorted_play():
    play_provider = PlayProvider(1)
    play_provider.add_landlord_id(2)
    play_hand, state_vector, action_list = play_provider.provide(
        np.array([1, 1, 1, 3, 3, 4, 4, 8, 8, 9, 9, 9, 10, 11, 13, 13, 14]), 15, 17)

    action = sample(action_list, 1)[0]
    print(PlayProvider.ActionProvider.ACTION_VIEW[action], execute_play(play_hand, action))

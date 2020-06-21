# -*- coding: utf-8 -*-
"""
执行动作的模块
@author 江胤佐
"""
from typing import Union, List

import numpy as np

from ai.provider import Hand
from card.combo import Combo


class Executor:
    """
    将状态转换为实际打出的牌的类
    @see provider
    """
    def __init__(self):
        self._action = None
        self._hand: Union[Hand, None] = None
        self._bad_hand = None

    @staticmethod
    def _pick(value, actions: List[np.ndarray]) -> np.ndarray:
        if value == 1:
            return actions[0]
        if value == 4:
            return actions[-1]
        if value == 2:
            return actions[len(actions) // 2]
        if value == 3:
            return actions[(len(actions) + 1) // 2]

    def _solo_execute(self) -> np.ndarray:
        """出单"""
        if self._action == 1:
            return np.array([self._hand.min_solo])
        if self._action == 51:
            return np.array([self._hand.max_solo])
        if self._hand.solo:
            return self._pick(self._action // 10, self._hand.solo)
        return self._pick(self._action // 10, self._bad_hand.solo)

    def _pair_execute(self) -> np.ndarray:
        """出对"""
        if self._hand.pair:
            return self._pick(self._action // 10, self._hand.pair)
        return self._pick(self._action // 10, self._bad_hand.pair)

    def execute(self, action: int, hand: Hand, bad_hand: Hand = None, last_combo: Combo = None) -> np.ndarray:

        # 空过
        if action == 0:
            return np.array([])
        self._hand = hand
        self._action = action
        self._bad_hand = bad_hand

        if action % 10 == 1:
            return self._solo_execute()
        elif action % 10 == 2:
            return self._pair_execute()

        return np.array([])

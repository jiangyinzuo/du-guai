# -*- coding: utf-8 -*-
"""
执行动作的模块
@author 江胤佐
"""
from typing import Union, List

import numpy as np

from ai.provider import Hand


class Executor:
    def __init__(self):
        self._result = None
        self._hand: Union[Hand, None] = None
        self._bad_hand = None

    @staticmethod
    def _pick(value, actions: List[np.ndarray]):
        if value == 1:
            return actions[0]
        if value == 4:
            return actions[-1]
        if value == 2:
            return actions[len(actions) // 2]
        if value == 3:
            return actions[(len(actions) + 1) // 2]

    def _solo_execute(self):
        if self._result == 1:
            return [self._hand.min_solo]
        if self._result == 51:
            return [self._hand.max_solo]
        if self._hand.solo:
            return self._pick(self._result // 10, self._hand.solo)
        return self._pick(self._result // 10, self._bad_hand.solo)

    def execute(self, result: int, hand: Hand, bad_hand: Hand = None) -> Union[np.ndarray, List[int]]:

        # 空过
        if result == 0:
            return []
        self._hand = hand
        self._result = result
        self._bad_hand = bad_hand
        if result % 10 == 1:
            return self._solo_execute()
        elif result % 10 == 2:
            pass

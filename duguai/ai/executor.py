# -*- coding: utf-8 -*-
"""
执行动作的模块
@author 江胤佐
"""
from typing import Union, List

import numpy as np

from ai.provider import Hand, Provider
from card import CARD_G1, CARD_G0
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
        self._last_combo_value: int = -1

    @staticmethod
    def _pick(value, actions: List[np.ndarray]) -> np.ndarray:
        if value == 1:
            return actions[0]
        if value == 4:
            return actions[-1]
        if value == 2:
            return actions[len(actions) // 2]
        if value == 3:
            if len(actions) == 1:
                return actions[0]
            return actions[(len(actions) + 1) // 2]

    def _solo_execute(self) -> np.ndarray:
        """出单"""
        if self._action == Provider.ActionProvider.MIN_SOLO:
            return np.array([self._hand.min_solo])
        if self._action == Provider.ActionProvider.MAX_SOLO:
            return np.array([self._hand.max_solo])
        if self._hand.solo:
            return self._pick(self._action // 10, self._hand.solo)
        return self._pick(self._action // 10, self._bad_hand.solo)

    def _pair_execute(self) -> np.ndarray:
        """出对"""
        if self._hand.pair:
            return self._pick(self._action // 10, self._hand.pair)
        return self._pick(self._action // 10, self._bad_hand.pair)

    def _trio_execute(self) -> np.ndarray:
        """出三带零/一/二"""
        if self._hand.trio:
            trio = self._pick(self._action // 100, self._hand.trio)
        else:
            trio = self._pick(self._action // 100, self._bad_hand.trio)

        # 三带一
        if self._action // 10 % 10 == 1:
            take = self._get_take(1, 1)
        # 三带二
        elif self._action // 10 % 10 == 2:
            take = self._get_take(2, 1)
        else:
            take = np.array([])
        return np.concatenate([trio, take])

    def _get_take(self, kind: int, length: int) -> np.ndarray:
        """获取带几的牌"""
        if length == 1:
            if kind == 1:
                return self._pick(1, self._hand.solo if self._hand.solo else self._bad_hand.solo)
            return self._pick(1, self._hand.pair if self._hand.pair else self._bad_hand.pair)

        takes = self._hand.solo if kind == 1 else self._hand.pair

        if len(takes) >= length:
            return np.partition(takes, length)[:length]
        elif self._last_combo_value != -1:
            bad_takes = self._bad_hand.solo if kind == 1 else self._bad_hand.pair
            length -= len(takes)
            return np.concatenate([takes, np.partition(bad_takes, length)[:length]])
        else:
            return np.array([])

    def _quadplex_or_bomb_execute(self) -> np.ndarray:
        """打出炸弹或四带二或王炸"""
        if self._action == Provider.ActionProvider.ROCKET:
            return np.array([CARD_G0, CARD_G1])
        elif self._action == Provider.ActionProvider.LITTLE_BOMB:
            return np.array(self._hand.bomb[0] if self._hand.bomb else self._bad_hand.bomb[0])
        elif self._action == Provider.ActionProvider.BIG_BOMB:
            return np.array(self._hand.bomb[-1] if self._hand.bomb else self._bad_hand.bomb[-1])

        if self._action == Provider.ActionProvider.FOUR_TAKE_ONE:
            take = self._get_take(1, 2)
        else:
            take = self._get_take(2, 2)

        for b in self._hand.bomb:
            if b[0] > self._last_combo_value:
                return np.concatenate([take, b])
        for b in self._bad_hand.bomb:
            if b[0] > self._last_combo_value:
                return np.concatenate([take, b])

        raise ValueError('找不到炸弹/四带二/王炸')

    def _seq_solo5_execute(self) -> np.ndarray:
        for seq in self._hand.seq_solo5:
            if seq[-1] > self._last_combo_value:
                return seq
        for seq in self._bad_hand.seq_solo5:
            if seq[-1] > self._last_combo_value:
                return seq
        raise ValueError('找不到长度为5的顺子')

    def _other_seq_execute(self) -> np.ndarray:
        if self._last_combo_value == -1:
            if self._hand.seq:
                return self._hand.seq[0]
            if self._hand.plane:
                takes = self._get_take(1, len(self._hand.plane[0]) // 3)
                if takes.size <= 0:
                    takes = self._get_take(2, len(self._hand.plane[0]) // 3)
                return np.concatenate([takes, self._hand.plane[0]])
        else:
            # TODO
            raise NotImplementedError('还没写完')

    def execute(self, action: int, hand: Hand, bad_hand: Hand = None, last_combo: Combo = None) -> np.ndarray:
        """
        给出要AI的决策对应实际要打出来的牌
        @param action: AI通过Q-learning算法给出的动作
        @param hand: 好的拆牌
        @param bad_hand: 差的拆牌
        @param last_combo: 上一次出牌的combo
        @return: 要打出的牌
        """

        # 空过
        if action == 0:
            return np.array([])
        self._hand = hand
        self._action = action
        self._bad_hand = bad_hand
        self._last_combo_value = last_combo.value if last_combo else -1

        execute_tuple = (self._solo_execute,
                         self._pair_execute,
                         self._trio_execute,
                         self._quadplex_or_bomb_execute,
                         self._seq_solo5_execute,
                         self._other_seq_execute)

        return execute_tuple[action % 10 - 1]()

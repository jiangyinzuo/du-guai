# -*- coding: utf-8 -*-
"""
给AI提供state和action的模块。该模块是对decompose的进一步处理
"""
from __future__ import annotations

from typing import List

import numpy as np

from ai.decompose import PlayDecomposer, FollowDecomposer
from card import CardsType, ActionsType, CARD_G1
from card.combo import Combo


class Actions:
    def __init__(self):
        self._solo: ActionsType = []
        self._pair: ActionsType = []
        self._trio: ActionsType = []
        self._bomb: ActionsType = []
        self._quadplex: ActionsType = []
        self._plane: ActionsType = []
        self._seq: ActionsType = []
        self._shuttle: ActionsType = []

    @property
    def solo(self) -> ActionsType:
        return self._solo

    @property
    def pair(self) -> ActionsType:
        return self._pair

    @property
    def trio(self) -> ActionsType:
        return self._trio

    @property
    def bomb(self) -> ActionsType:
        return self._bomb

    @property
    def quadplex(self) -> ActionsType:
        return self._quadplex

    @property
    def plane(self) -> ActionsType:
        return self._plane

    @property
    def seq(self) -> ActionsType:
        return self._seq

    @property
    def shuttle(self) -> ActionsType:
        return self._shuttle


class Provider:

    _LAND_LORD = 0
    _FARMER_1 = 1
    _FARMER_2 = 2

    def __init__(self, land_lord_id: int, player_id: int):
        self.__land_lord_id = land_lord_id
        self.__player_id = player_id
        self._action_provider = Provider.ActionProvider(self)
        self._state_provider = Provider.StateProvider(self)

    @property
    def identity(self) -> int:
        """
        玩家身份
        @return: 0: 地主; 1: 地主下家农民; 2: 地主上家农民
        """
        return (self.__player_id - self.__land_lord_id + 3) % 3

    def provide(self, hand: CardsType, hand_p: int, hand_n: int, last_combo: Combo = None):
        if last_combo is None:
            actions = self._action_provider.provide_play(hand, hand_p, hand_n)
        else:
            actions = self._action_provider.provide_follow(hand, last_combo, hand_p, hand_n)

    class ActionProvider:
        """
        AI出牌或跟牌时，给AI提供动作的类。
        动作被化简为以下几个：
        0：空过
        x1：出单，x取【0-3】, 表示出最小、偏小、偏大、最大的单
        x2：出对，x取【0-3】, 表示出最小、偏小、偏大、最大的对
        x3：出三，x取【0-2】, 表示出最小、中间、最大的三
        x4：出长度为5的顺子，x取【0-1】，表示出较小的或较大的顺子
        x5：出一个小的炸弹, x取【0-1】，表示出小炸弹或大炸弹
        6：出其它各种飞机连对顺子
        """

        def __init__(self, outer: Provider):
            self.__outer = outer
            self._play_decomposer: PlayDecomposer = PlayDecomposer()
            self._follow_decomposer: FollowDecomposer = FollowDecomposer()
            self.__actions: Actions

        def __part_action(self, hand: CardsType) -> None:
            d_actions: List[np.ndarray] = self._play_decomposer.get_good_plays(hand)
            self.__actions: Actions = Actions()
            for a in d_actions:
                if len(a) == 1:
                    self.__actions.solo.append(a)
                elif len(a) == 2:
                    if a[-1] != CARD_G1:
                        self.__actions.pair.append(a)
                    else:
                        self.__actions.bomb.append(a)
                elif len(a) == 3:
                    self.__actions.trio.append(a)
                elif len(a) == 4:
                    self.__actions.bomb.append(a)
                elif len(a) % 4 == 0 and a[0] == a[3]:
                    self.__actions.shuttle.append(a)
                elif len(a) % 3 == 0 and a[0] == a[2] and a[0] != a[3]:
                    self.__actions.plane.append(a)
                else:
                    self.__actions.seq.append(a)

        def __combine_and_divide(self, hand_p: int, hand_n: int) -> None:

            if self.__outer.identity == Provider._LAND_LORD:
                min_hand = min(hand_p, hand_n)
                if min_hand == 2 and len(self.__actions.solo) == 0 and len(self.__actions.pair) > 0:
                    self.__actions.solo.append(self.__actions.pair[0][0])

        def provide_play(self, hand: CardsType, hand_p: int, hand_n: int) -> Actions:
            self.__part_action(hand)
            self.__combine_and_divide(hand_p, hand_n)
            return self.__actions

        def provide_follow(self, hand: CardsType, combo: Combo, hand_p: int, hand_n: int) -> Actions:
            pass

    class StateProvider:
        """
        AI出牌或跟牌时，给AI提供状态的类
        状态是一个长度为13的特征向量。特征的含义以及取值范围如下：
        （备注：为减少取值范围，平均值计算均为整除）

        状态说明               属性名                 取值范围（均为整数）
        ------------------------------------------------------------
        f1_min(单张)          solo_min              [0, 3]
        f1_max(单张)          solo_max              [0, 3]
        f2_min(对子)          pair_min              [0, 2]
        f2_max(对子)          pair_max              [0, 2]
        f2_min(三)            trio_min              [0, 2]
        f2_max(三)            trio_max              [0, 2]
        单顺（长为5）/5        seq_solo_5             [0, 2]
        存在其它牌型            other_seq_count       [0, 1]
        炸弹数量（大于2记作2）    bomb_count            [0, 2]
        是否有王炸              rocket                [0, 1]
        玩家位置                player               [0, 2]
        上家手牌数-1            hand_p(大于5都记作5)    [0, 5]
        下家手牌数-1            hand_n(大于5都记作5)    [0, 5]
        ------------------------------------------------------------

        f1_min(x) = switch mean(x[:2]):  [1, 4] -> 0; [5, 8] -> 1; [9,  12] -> 2; [13, 15] -> 3
        f1_max(x) = switch max(x):       [1, 4] -> 0; [5, 8] -> 1; [9,  12] -> 2; [13, 15] -> 3
        f2_min(x) = switch mean(x[:2]):  [1, 5] -> 0; [6,10] -> 1; [11, 13] -> 2
        f2_max(x) = switch max(x):       [1, 5] -> 0; [6,10] -> 1; [11, 13] -> 2
        总共有 (4+3+2+1)*(3+2+1)^2*3*2*3*2*3*6*6 = 1399680 种状态

        出牌和跟牌的state一样，区别只在于跟牌的时候部分action不能走
        """
        STATE_LEN = 13

        def __init__(self):
            self._actions = None
            self._state = None

        def provide(self, actions: List[np.ndarray]):
            self._actions = actions
            combo = Combo()
            for a in actions:
                combo.cards = a

        @property
        def state(self):
            """
            被包装的状态数组
            """
            return self._state

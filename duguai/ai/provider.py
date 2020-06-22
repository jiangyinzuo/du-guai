# -*- coding: utf-8 -*-
"""
给AI提供state和action的模块。该模块是对decompose的进一步处理
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ai.decompose import PlayDecomposer, FollowDecomposer
from card import CARD_G1, CARD_6
from card.combo import Combo


class Hand:
    """
    对手牌进行进一步分类的类
    """

    def __init__(self, cards: np.ndarray, d_actions, last_combo: Combo):
        """
        初始化Hand类
        @param cards: 玩家的初始手牌
        @param d_actions: 经过初步分解后的行动。（初步分解在decompose.py模块下）
        @param last_combo: 上一个出牌的Combo
        @see decompose.py
        """
        self._cards: np.ndarray = cards
        self._solo: List[np.ndarray] = []
        self._pair: List[np.ndarray] = []
        self._trio: List[np.ndarray] = []
        self._bomb: List[np.ndarray] = []
        self._plane: List[np.ndarray] = []
        self._seq_solo5: List[np.ndarray] = []
        self._other_seq: List[np.ndarray] = []
        self._has_rocket: bool = False
        self._min_solo = None
        self._max_solo = None
        self._last_combo = last_combo

        self.__part_action(d_actions)
        self.find_valid_plane()

    def __part_action(self, d_actions: List[np.ndarray]) -> None:
        """对d_action进行初步分类"""

        for a in d_actions:

            # 含王炸
            if len(a) == 2 and a[-1] == CARD_G1:
                self._has_rocket = True

            # 分类单牌
            if 1 <= len(a) <= 4:
                (self.solo, self.pair, self.trio, self.bomb)[len(a) - 1].append(a)

            # 分类长度为5的单顺
            elif len(a) == 5 and a[0] == a[1] - 1 == a[3] - 3 == a[4] - 4:
                self.seq_solo5.append(a)

            # 分类飞机
            elif len(a) % 3 == 0 and a[0] == a[2] and a[0] != a[3]:
                self.plane.append(a)

            # 分类其它顺子
            else:
                self.other_seq.append(a)

    def find_valid_other_seq(self):
        """
        找到合法的顺子/连对
        """

    def find_valid_plane(self):
        """
        找到合法的飞机
        """

    @property
    def cards(self) -> np.ndarray:
        """玩家原始的卡牌"""
        return self._cards

    @property
    def solo(self) -> List[np.ndarray]:
        """单"""
        return self._solo

    @property
    def pair(self) -> List[np.ndarray]:
        """对"""
        return self._pair

    @property
    def trio(self) -> List[np.ndarray]:
        """三"""
        return self._trio

    @property
    def bomb(self) -> List[np.ndarray]:
        """炸弹"""
        return self._bomb

    @property
    def plane(self) -> List[np.ndarray]:
        """飞机"""
        return self._plane

    @property
    def other_seq(self) -> List[np.ndarray]:
        """其它各种序列"""
        return self._other_seq

    @property
    def seq_solo5(self) -> List[np.ndarray]:
        """长度为5的单顺"""
        return self._seq_solo5

    @property
    def has_rocket(self) -> bool:
        """是否有王炸"""
        return self._has_rocket

    @property
    def min_solo(self) -> int:
        """强拆的最小单牌"""
        return self._min_solo

    @property
    def max_solo(self) -> int:
        """强拆的最大单牌"""
        return self._max_solo

    @max_solo.setter
    def max_solo(self, value):
        self._max_solo = value

    @min_solo.setter
    def min_solo(self, value):
        self._min_solo = value


class Provider:
    """
    给AI提供state和action的类
    @note 一个AI在整副对局中只创建并使用一个Provider对象
    """
    _LAND_LORD = 0
    _FARMER_1 = 1
    _FARMER_2 = 2

    def __init__(self, player_id: int):
        self.__landlord_id = None
        self.__player_id = player_id
        self._action_provider = Provider.ActionProvider(self)
        self._state_provider = Provider.StateProvider(self)

    def add_landlord(self, landlord_id: int):
        """
        设置地主
        @param landlord_id: 地主玩家的id
        """
        self.__landlord_id = landlord_id

    @property
    def identity(self) -> int:
        """
        玩家身份
        @return: 0: 地主; 1: 地主下家农民; 2: 地主上家农民
        """
        return (self.__player_id - self.__landlord_id + 3) % 3

    def provide(self, cards: np.ndarray, hand_p: int, hand_n: int, last_combo_owner: int, last_combo: Combo = None) \
            -> Tuple[np.ndarray, List[int], Hand, Hand]:
        """
        提供状态和动作
        @param cards: 当前玩家手牌
        @param hand_p: 上一个玩家手牌数量
        @param hand_n: 下一个玩家手牌数量
        @param last_combo_owner: 上一个出牌的玩家id
        @param last_combo: 上一个要跟的Combo
        @return: 元组，(state, action, good_hand, bad_hand)
        """
        if last_combo is None:
            bad_hand = None
            good_hand, action_list = self._action_provider.provide_play(cards, hand_p, hand_n)
        else:
            good_hand, bad_hand, action_list = self._action_provider.provide_follow(cards, last_combo)

        state: np.ndarray = self._state_provider.provide(good_hand, hand_p, hand_n,
                                                         (last_combo_owner - self.__landlord_id + 3) % 3)
        return state, action_list, good_hand, bad_hand

    class ActionProvider:
        """
        AI出牌或跟牌时，给AI提供动作的类。
        动作被化简为以下几个：
        0：空过
        x1：出单，x取【0-5】, 表示出 强行最小、最小、中、偏大、最大、强行最大的单
        x2：出对，x取【1-4】, 表示出 最小、中、偏大、最大的对
        xy3：出三，x取【0-2】, 表示出最小、中间、最大的三；y表示带单还是带对
        xy4：出一个小的炸弹, x取【0-2】，表示出小炸弹或大炸弹或王炸, y表示带单还是带双还是不带
        x5：出长度为5的顺子，x取【0-1】，表示出较小的或较大的顺子
        6：出其它连对顺子
        7：出飞机
        """

        PASS = 0
        MIN_SOLO = 1
        MAX_SOLO = 51
        PAIRS = [12, 22, 32, 42]

        LITTLE_BOMB = 4
        BIG_BOMB = 104
        ROCKET = 204
        FOUR_TAKE_ONE = 14
        FOUR_TAKE_TWO = 24

        OTHER_SEQ = 6
        PLANE = 7

        def __init__(self, outer: Provider):
            self._outer: Provider = outer
            self._play_decomposer: PlayDecomposer = PlayDecomposer()
            self._follow_decomposer: FollowDecomposer = FollowDecomposer()

            # 若玩家为1号农民，当2号农民只剩一张牌时，仅尝试拆一次单牌。
            self._split_min_solo: bool = True
            self.action_list: List[int] = []

        def _split_solo(self, cards: np.ndarray, hand: Hand):
            self._split_min_solo = False
            min_card = np.min(cards)
            if min_card < CARD_6:
                hand.min_solo = min_card
            else:
                pair_card, trio_card = -1, -1
                if len(hand.pair):
                    pair_card: int = hand.pair[0][0]
                if len(hand.trio):
                    trio_card: int = hand.trio[0][0]
                min_card: int = min(pair_card, trio_card)
                if min_card != -1:
                    hand.min_solo = min_card

        @classmethod
        def _add_solo_actions(cls, hand, action_list):
            if hand.min_solo:
                action_list.append(cls.MIN_SOLO)
            if len(hand.solo):
                action_list.extend([11, 21, 31, 41])
            if hand.max_solo:
                action_list.append(cls.MAX_SOLO)

        @classmethod
        def _add_bomb_actions(cls, hand, action_list):
            if hand.has_rocket:
                action_list.append(cls.ROCKET)
            if len(hand.bomb):
                if len(hand.bomb) == 1:
                    action_list.append(cls.LITTLE_BOMB)
                else:
                    action_list.extend([cls.LITTLE_BOMB, cls.BIG_BOMB])
                if len(hand.solo) >= 2 or len(hand.pair):
                    action_list.append(cls.FOUR_TAKE_ONE)
                if len(hand.pair) >= 2:
                    action_list.append(cls.FOUR_TAKE_TWO)

        @staticmethod
        def _has_take(hand, take_kind: int, take_num: int) -> bool:
            if take_kind == 0:
                return True
            if take_kind == 1:
                return len(hand.solo) + len(hand.pair) * 2 >= take_num
            if take_kind == 2:
                return len(hand.pair) >= take_num
            raise ValueError('非法的值')

        def _add_actions(self, hand: Hand, last_combo: Combo = None):
            action_list = []
            if last_combo is None or last_combo.is_solo():
                self._add_solo_actions(hand, action_list)
            if (last_combo is None or last_combo.is_pair()) and hand.pair:
                action_list.extend(self.PAIRS)
            if hand.trio:
                action_list.extend([3, 103, 103])
                if hand.solo:
                    action_list.extend([13, 113, 123])
                if hand.pair:
                    action_list.extend([23, 123, 223])
            if hand.seq_solo5:
                action_list.extend([5, 15])
            self._add_bomb_actions(hand, action_list)

            # 找到合法的序列并加入到action_list中
            hand.find_valid_other_seq(last_combo)
            hand.find_valid_plane(last_combo)
            return action_list

        def provide_play(self, cards: np.ndarray, hand_p: int, hand_n: int) -> \
                Tuple[Hand, List[int]]:
            """
            提供出牌时候的actions
            @param cards: 玩家当前的手牌
            @param hand_p: 上一个玩家的手牌数
            @param hand_n: 下一个玩家的手牌数
            """
            self.action_list: List[int] = []
            d_actions = self._play_decomposer.get_good_plays(cards)
            hand: Hand = Hand(cards, d_actions)

            # 一农民剩2张，地主怕出对，又没有合适的单牌，故拆出一张单牌
            if self._outer.identity == Provider._LAND_LORD:
                min_hand = min(hand_p, hand_n)
                if min_hand == 2 and len(hand.solo) == 0 and len(hand.pair):
                    hand._min_solo = hand.pair[0][0]
            elif hand_n == 1 and hand_p == 1:
                hand.max_solo = np.max(cards)
            # 一号农民拆单牌
            elif self._outer.identity == Provider._FARMER_1 and hand_n == 1 and self._split_min_solo:
                self._split_solo(cards, hand)

            return hand, self._add_actions(hand)

        def provide_follow(self, cards: np.ndarray, last_combo: Combo) \
                -> Tuple[Hand, List[int]]:
            """
            提供跟牌时候的action
            @param cards:
            @param last_combo: 上一个出牌的Combo
            """
            self.action_list: List[int] = []
            d_actions = self._follow_decomposer.get_good_follows(cards, last_combo)
            good_hand: Hand = Hand(cards, d_actions)
            action_list = [self.PASS]
            action_list.extend(self._add_actions(good_hand, last_combo))
            return good_hand, list(set(action_list))

    class StateProvider:
        """
        AI出牌或跟牌时，给AI提供状态的类
        状态是一个长度为13的特征向量。特征的含义以及取值范围如下：
        （备注：// 表示整除）

        状态说明               属性名                 取值范围（均为整数）
        ------------------------------------------------------------
        f1_min(单张)          solo_min              [0, 3]
        f1_max(单张)          solo_max              [0, 3]
        f2_min(对子)          pair_min              [0, 2]
        f2_max(对子)          pair_max              [0, 2]
        f2_min(三)            trio_min              [0, 2]
        f2_max(三)            trio_max              [0, 2]
        最大单顺（长为5） // 5   seq_solo_5             [0, 2]
        存在其它牌型(不含4带2)   other_seq_count       [0, 1]
        炸弹数量（大于2记作2）    bomb_count            [0, 2]
        是否有王炸              rocket                [0, 1]
        玩家位置                player               [0, 2]
        上一个牌是谁打的         last_combo_owner      [0, 2]
        上家手牌数-1            hand_p(大于5都记作5)    [0, 5]
        下家手牌数-1            hand_n(大于5都记作5)    [0, 5]
        ------------------------------------------------------------

        f1_min(x) = switch mean(x[:2]):  [1, 4] -> 0; [5, 8] -> 1; [9,  12] -> 2; [13, 15] -> 3
        f1_max(x) = switch max(x):       [1, 4] -> 0; [5, 8] -> 1; [9,  12] -> 2; [13, 15] -> 3
        f2_min(x) = switch mean(x[:2]):  [1, 5] -> 0; [6,10] -> 1; [11, 13] -> 2
        f2_max(x) = switch max(x):       [1, 5] -> 0; [6,10] -> 1; [11, 13] -> 2
        总共有 (4+3+2+1)*(3+2+1)^2*3*2*3*2*3*3*6*6 = 4199040 种状态
        """
        STATE_LEN = 14

        @staticmethod
        def __value_to_f1(value):
            if value <= 4:
                return 0
            elif value <= 8:
                return 1
            elif value <= 12:
                return 2
            else:
                return 3

        @staticmethod
        def __value_to_f2(value):
            if value <= 5:
                return 0
            elif value <= 10:
                return 1
            else:
                return 3

        @classmethod
        def _f_min(cls, solos: List[np.ndarray], t: int = 1) -> int:
            if len(solos) == 1:
                value = solos[0][0]
            else:
                value = np.mean(np.partition(np.array(solos).ravel(), 1)[0:2])
            return cls.__value_to_f1(value) if t == 1 else cls.__value_to_f2(value)

        @classmethod
        def _f_max(cls, solos: List[np.ndarray], t: int = 1) -> int:
            value = np.max(solos)
            return cls.__value_to_f1(value) if t == 1 else cls.__value_to_f2(value)

        def __init__(self, outer: Provider):
            self._outer = outer

        def provide(self, hand: Hand, hand_p: int, hand_n: int, last_combo_owner: int) -> np.ndarray:
            """
            为AI提供状态
            @param hand: 玩家的手牌
            @param last_combo_owner: 上一个手牌是谁打的
            @param hand_p: 上一个玩家的手牌数量
            @param hand_n: 下一个玩家的手牌数量
            @return: 长度为 STATE_LEN 的特征向量
            """
            state_vector = np.zeros(Provider.StateProvider.STATE_LEN, dtype=int)
            if len(hand.solo):
                state_vector[0:2] = self._f_min(hand.solo), self._f_max(hand.solo)
            if len(hand.pair):
                state_vector[2:4] = self._f_min(hand.pair, 2), self._f_max(hand.pair, 2)
            if len(hand.trio):
                state_vector[4:6] = self._f_min(hand.pair, 2), self._f_max(hand.pair, 2)
            if len(hand.seq_solo5):
                state_vector[6] = np.max(hand.seq_solo5)
            if len(hand.other_seq) > 0 or len(hand.plane) > 0:
                state_vector[7] = 1
            bomb_count = 2 if len(hand.bomb) > 2 else len(hand.bomb)
            state_vector[8] = bomb_count
            state_vector[9] = hand.has_rocket
            state_vector[10:14] = self._outer.identity, last_combo_owner, hand_p, hand_n
            return state_vector

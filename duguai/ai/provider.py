# -*- coding: utf-8 -*-
"""
给AI提供state和action的模块。该模块是对decompose的进一步处理
"""
from typing import List

from ai.decompose import PlayDecomposer, FollowDecomposer
from card import CardsType
from card.combo import Combo


class ActionProvider:
    """
    AI出牌或跟牌时，给AI提供动作的类
    """

    def __init__(self):
        play_decomposer: PlayDecomposer = PlayDecomposer()
        follow_decomposer: FollowDecomposer = FollowDecomposer()

    def provide_play(self, hand: CardsType):
        pass

    def provide_follow(self, hand: CardsType, combo: Combo):
        pass


class StateProvider:
    """
    AI出牌或跟牌时，给AI提供状态的类
    """
    pass


class StateWrapper:
    """
    状态包装类
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

    def __init__(self, state: List[int]):
        if len(state) != StateWrapper.STATE_LEN:
            raise ValueError('state长度必须为13')
        self._state: List[int] = state

    @property
    def state(self):
        """
        被包装的状态数组
        """
        return self._state

    @state.setter
    def state(self, value: List[int]):
        if len(value) != StateWrapper.STATE_LEN:
            raise ValueError('state长度必须为13')
        self._state = value

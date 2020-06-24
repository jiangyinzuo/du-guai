# -*- coding: utf-8 -*-
"""
游戏运行环境
@author: 江胤佐
"""
from __future__ import annotations

import abc
from functools import wraps
from typing import List, Iterator, Union, Tuple

import numpy as np

from card.cards import cards_view
from card.combo import Combo
from duguai import mode
from utils import is_in

SPLIT_LINE = '----------------------------------------'


def _remove_last_combo(func):
    """
    从手牌中移除打出的牌的装饰器
    """

    @wraps(func)
    def decorated(player: GameEnv.AbstractPlayer):
        """
        玩家出牌后，删去玩家手牌中出牌的卡牌
        """
        result = func(player)
        hand = list(player.hand)
        for i in player.last_combo.cards:
            hand.remove(i)
        player.hand = np.array(hand)
        return result

    return decorated


class GameEnv:
    """
    游戏运行环境类。该类包括：
    1. 所有游戏运行的变量，如出牌轮到谁，每个玩家当前拥有的牌，地主获得的牌等；
    2. 一些游戏流程的方法，如洗牌。
    """

    U_LANDLORD = 'update_landlord'
    U_GAME_OVER = 'update_game_over'
    U_LAST_COMBO = 'update_last_combo'
    U_MSG = 'update_msg'

    class AbstractPlayer(metaclass=abc.ABCMeta):
        """
        玩家抽象类。定义了玩家叫地主、跟牌、先手出牌方法。
        """

        def __init__(self, game_env: GameEnv, order: int):
            self.game_env = game_env
            self.order = order
            self.last_combo: Combo = Combo()

        @property
        def hand(self) -> np.ndarray:
            """
            玩家当前的手牌。
            @return: numpy数组，按从小到大排列
            """
            return self.game_env.cards[self.order]

        @hand.setter
        def hand(self, v):
            self.game_env.cards[self.order] = v

        @abc.abstractmethod
        def call_landlord(self) -> bool:
            """
            叫地主
            :return: True: 叫; False: 不叫
            """
            pass

        @abc.abstractmethod
        def update_landlord(self, landlord_id: int) -> None:
            """
            由GameEnv通知各个玩家谁是地主。
            @param landlord_id: 地主玩家的id
            """
            pass

        @abc.abstractmethod
        def update_msg(self, msgs: Union[Iterator, str]) -> None:
            """GameEnv通知玩家字符串消息"""
            pass

        @abc.abstractmethod
        def update_last_combo(self, is_play: bool) -> None:
            """GameEnv通知出牌"""
            pass

        @abc.abstractmethod
        def update_game_over(self, victor: Union[Tuple[int], int]) -> None:
            """
            GameEnv通知游戏结束
            @param victor: 胜利者
            """
            pass

        @abc.abstractmethod
        def follow(self) -> None:
            """
            跟牌
            """
            pass

        @abc.abstractmethod
        def play(self) -> None:
            """
            先手出牌
            """
            pass

        def valid_follow(self) -> bool:
            """
            判断跟牌是否合法
            @return: 合法：True；非法：False
            """
            return self.last_combo.cards.size == 0 or is_in(
                self.last_combo.cards,
                self.hand
            ) and self.last_combo.is_valid() and self.last_combo > self.game_env.last_combo

    def __init__(self):
        self.cards: List[np.ndarray] = []

        # 玩家数组
        self._players: List[GameEnv.AbstractPlayer] = []
        self.turn: int
        self.land_lord: int
        self._last_combo_owner: int
        self._former_combo: Combo

    def _init(self):

        # 卡牌二维数组, 前3个代表玩家0、1、2的初始手牌（各17张）最后一项代表3张地主牌
        self.cards: List[np.ndarray] = np.split(np.asarray([card for card in range(1, 14)] * 4 + [14, 15], dtype=int),
                                                [17, 34, 51])

        # 当前轮到第几个玩家
        self.turn: int = 0

        # 地主编号
        self.land_lord: int = -1

        # 上一个combo是谁打的。若上一个combo是自己打的，则出牌；否则跟牌。游戏开始时为地主玩家的id
        self._last_combo_owner: int = -1

        # 上一个Combo（除了初始状态，不含打牌过程中产生的PASS）
        self._former_combo: Combo = Combo()

    def add_players(self,
                    p1: GameEnv.AbstractPlayer,
                    p2: GameEnv.AbstractPlayer,
                    p3: GameEnv.AbstractPlayer):
        """
        实例化PlayEnv类后，需要添加玩家
        @param p1: 玩家0
        @param p2: 玩家1
        @param p3: 玩家2
        """
        self._players = [p1, p2, p3]

    def notify(self, func_name, *args, **kwargs):
        """
        游戏环境通知各个玩家
        @param func_name: 通知的函数
        @param args: 传给玩家对象的参数
        @param kwargs: 传给玩家对象的关键字参数
        """
        for player in self._players:
            player.__getattribute__(func_name)(*args, **kwargs)

    def _start_msg(self):
        yield '斗地主开始！'
        if mode == 'debug':
            yield '*** 调试模式下可以预先看牌 ***'
            yield SPLIT_LINE
            i = 0
            for p in self._players:
                yield '玩家%d的手牌: ' % i
                yield cards_view(p.hand)
                i += 1
            yield SPLIT_LINE

    def start(self) -> None:
        """
        开始游戏
        """
        self._init()
        assert len(self._players) == 3, '开始游戏前先添加玩家'
        self.__call_landlord()

        self.notify(GameEnv.U_MSG, msgs=self._start_msg())

        self.__round_robin()

    def shuffle(self) -> None:
        """
        洗牌
        """
        self.cards = np.concatenate(self.cards, axis=0)
        np.random.shuffle(self.cards)
        self.cards = np.split(self.cards, [17, 34, 51])
        for c in self.cards:
            c.sort()

    @property
    def last_combo(self):
        """
        上一个打出的手牌组合（不包括玩家空过）
        """
        return self._former_combo

    @property
    def hand_p(self) -> int:
        """当前玩家上家的手牌数量"""
        return len(self.cards[(self.turn + 2) % 3])

    @property
    def hand_n(self) -> int:
        """当前玩家下家的手牌数量"""
        return len(self.cards[(self.turn + 1) % 3])

    @property
    def last_combo_owner_id(self) -> int:
        """
        上一个牌是哪个玩家打的
        @return: 玩家id
        """
        return self._last_combo_owner

    def user_info(self, player: int) -> str:
        """
        当前打牌的身份与玩家id
        @param player 0：当前玩家， 1：下家， -1：上家
        """
        return ('[地主' if self.land_lord == (self.turn + player + 3) % 3 else '[农民') + ' 玩家%d] ' % self.turn

    def __call_landlord(self):

        self.notify(GameEnv.U_MSG, msgs='进入叫地主环节')
        while self.land_lord == -1:
            self.shuffle()
            for player in self._players:

                if player.call_landlord():
                    self.land_lord = self.turn
                    self.notify(GameEnv.U_LANDLORD, landlord_id=self.turn)

                    self.cards[self.turn] = np.concatenate([self.cards[self.turn], self.cards[3]])
                    self.cards[self.turn].sort()
                    self._last_combo_owner = self.turn
                    break

                self.turn = (self.turn + 1) % 3

    def __round_robin(self):
        while True:
            if self._last_combo_owner == self.turn:
                self.notify(GameEnv.U_MSG, msgs=self.user_info(0) + '出牌')
                self._players[self.turn].play()
                is_play = True
            else:
                self.notify(
                    GameEnv.U_MSG,
                    msgs=self.user_info(0) + '跟牌(先前的牌由 玩家{} 打出 {})'.format(
                        self._last_combo_owner, self._former_combo.cards_view)
                )
                self._players[self.turn].follow()
                is_play = False

            if self._players[self.turn].last_combo.is_not_empty():
                self._last_combo_owner = self.turn
                self._former_combo = self._players[self.turn].last_combo
            self.notify(GameEnv.U_LAST_COMBO, is_play=is_play)

            if self.cards[self.turn].size == 0:
                if self.land_lord == self.turn:
                    self.notify(GameEnv.U_GAME_OVER, victor=self.land_lord)
                else:
                    farmers = {0, 1, 2}
                    farmers.remove(self.land_lord)
                    self.notify(GameEnv.U_GAME_OVER, victor=farmers)
                return

            self.turn = (self.turn + 1) % 3

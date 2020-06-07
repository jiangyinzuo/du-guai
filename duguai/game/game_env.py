# -*- coding: utf-8 -*-
"""
游戏运行环境
@author: 江胤佐
"""
from __future__ import annotations

import abc
from typing import List

import numpy as np

from algorithm.call_landlord import get_svc, process, z_score
from duguai import mode
from game.cards import cards_view, Combo


class GameEnv:
    """
    游戏运行环境类。该类包括：
    1. 所有游戏运行的变量，如出牌轮到谁，每个玩家当前拥有的牌，地主获得的牌等；
    2. 一些游戏流程的方法，如洗牌。
    """

    class AbstractPlayer(metaclass=abc.ABCMeta):
        """
        玩家抽象类。定义了玩家叫地主、跟牌、先手出牌方法。
        """

        def __init__(self, game_env: GameEnv, order: int):
            self.game_env = game_env
            self.order = order
            self.last_combo: Combo = Combo()

        @property
        def cards(self) -> np.ndarray:
            """
            玩家当前的手牌。
            @return: numpy数组，按从小到大排列
            """
            return self.game_env.cards[self.order]

        @abc.abstractmethod
        def call_landlord(self) -> bool:
            """
            叫地主
            :return: True: 叫; False: 不叫
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

    def __init__(self):

        # 卡牌二维数组, 前3个代表玩家0、1、2的初始手牌（各17张）最后一项代表3张地主牌
        self.cards: List[np.ndarray] = np.split(np.asarray([card for card in range(1, 14)] * 4 + [14, 15], dtype=int),
                                                [17, 34, 51])

        # 玩家数组
        self.players: List[GameEnv.AbstractPlayer] = []

        # 当前轮到第几个玩家
        self.turn: int = 0

        # 地主编号
        self.land_lord: int = -1

        # 上一个combo是谁打的。若上一个combo是自己打的，则出牌；否则跟牌
        self.last_combo_owner: int = -1

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
        self.players = [p1, p2, p3]

    def start(self) -> None:
        """
        开始游戏
        """
        assert len(self.players) == 3, "开始游戏前先添加玩家"
        self.__call_landlord()
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
    def former_combo(self):
        return self._former_combo

    def __call_landlord(self):
        print('进入叫地主环节')
        while self.land_lord == -1:
            self.shuffle()
            for player in self.players:
                if player.call_landlord():
                    self.land_lord = self.turn
                    print('玩家{}叫了地主'.format(self.turn))
                    print("地主将获得的3张牌: {}".format(cards_view(self.cards[3])))
                    self.cards[self.turn] = np.concatenate([self.cards[self.turn], self.cards[3]])
                    break
                self.turn = (self.turn + 1) % 3
        self.cards[self.turn] = np.append(self.cards[self.turn], self.cards[3])
        self.last_combo_owner = self.turn

    def __round_robin(self):
        while True:
            if self.last_combo_owner == self.turn:
                self.players[self.turn].play()
            else:
                self.players[self.turn].follow()
            if len(self.cards[self.turn]) == 0:
                if self.land_lord == self.turn:
                    print('地主(玩家{})获胜'.format(self.land_lord))
                else:
                    farmers = {0, 1, 2}
                    farmers.remove(self.land_lord)
                    print('农民(玩家{}、玩家{})获胜'.format(farmers.pop(), farmers.pop()))
                break


class Human(GameEnv.AbstractPlayer):
    """
    人类玩家，由控制台输入输出进行操作
    @author 江胤佐
    """

    def __init__(self, game_env: GameEnv, order: int):
        super().__init__(game_env, order)

    def call_landlord(self) -> bool:
        """
        玩家叫地主
        @return: 叫: True; 不叫: False
        """
        print('玩家{}的手牌:'.format(self.order), cards_view(self.cards))
        return input('输入1叫地主, 输入其它键不叫地主') == '1'

    def follow(self) -> None:
        """
        玩家跟牌
        """
        while True:
            cards_v = input('输入要出的牌，以空格分隔。直接回车代表不出牌')
            self.last_combo.cards_view = cards_v
            if self.last_combo.is_valid() and self.last_combo > self.game_env.former_combo:
                break
            else:
                print('输入非法!')
        # TODO

    def play(self) -> None:
        pass


class Robot(GameEnv.AbstractPlayer):
    """
    AI，由机器学习算法决定操作
    @author 江胤佐
    """

    def __init__(self, game_env: GameEnv, order: int):
        super().__init__(game_env, order)
        self.svc = get_svc()

    def call_landlord(self) -> bool:
        """
        AI叫地主
        @return: 叫: True; 不叫: False
        """
        if mode == 'debug':
            print('AI{}的手牌:'.format(self.order), cards_view(self.cards))
        return self.svc.predict(z_score([process(self.cards), ])) == 1

    def follow(self) -> None:
        pass

    def play(self) -> None:
        pass

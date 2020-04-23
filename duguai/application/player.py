# -*- coding: utf-8 -*-
"""
玩家模块
"""

import abc

import numpy as np


class Player(metaclass=abc.ABCMeta):
    """
    玩家抽象类
    """

    def __init__(self, cards: np.ndarray, name: str):
        self.__order = None
        self.__cards = cards
        self.__name = name

    @abc.abstractmethod
    def call_landlord(self) -> bool:
        """
        叫地主
        :return: True: 叫; False: 不叫
        """
        pass

    @property
    def order(self):
        return self.__order

    @property
    def name(self):
        return self.__name

    @order.setter
    def order(self, order):
        self.__order = order


class Robot(Player):

    def call_landlord(self) -> bool:
        # TODO Robot叫地主
        return False


class Human(Player):

    def call_landlord(self) -> bool:
        return input('1: 叫地主; 其它任意输入: 不叫') == '1'

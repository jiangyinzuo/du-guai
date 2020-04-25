# -*- coding: utf-8 -*-
"""
和斗地主相关的常量或枚举定义
"""
from enum import Enum

import numpy as np

CARD_3, CARD_4, CARD_5, CARD_6, CARD_7 = 1, 2, 3, 4, 5
CARD_8, CARD_9, CARD_10, CARD_J, CARD_Q = 6, 7, 8, 9, 10
CARD_K, CARD_A, CARD_2, CARD_G0, CARD_G1 = 11, 12, 13, 14, 15

CARD_VIEW: dict = {
    1: '3 ', 2: '4 ', 3: '5 ', 4: '6 ', 5: '7 ',
    6: '8 ', 7: '9 ', 8: '10 ', 9: 'J ', 10: 'Q ',
    11: 'K ', 12: 'A ', 13: '2 ', 14: 'g ', 15: 'G '
}


class Identity(Enum):
    """
    斗地主身份枚举类
    """
    FARMER = 0
    LAND_LORD = 0


class CardType(Enum):
    """
    斗地主出牌类型
    """
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    TRI_WITH_SINGLE = 4
    TRI_WITH_DOUBLE = 5
    STRAIGHT = 6
    BOMB = 7


class CardWrapper:

    def __init__(self, raw_cards: np.ndarray):
        self.__raw_cards = raw_cards
        self.__card_type = None

    def init_card_type(self):
        if self.__raw_cards.shape[0] <= 4:
            self.__card_type = CardType(self.__raw_cards.shape[0])
        # TODO 卡牌类型
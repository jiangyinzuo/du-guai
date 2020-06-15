# -*- coding: utf-8 -*-
"""
和斗地主相关的常量或枚举定义
@author: 江胤佐
"""
from __future__ import annotations

from card import *
from duguai.card.card_helper import card_to_di


def cards_view(cards: np.ndarray) -> str:
    """
    获取牌的字符串形式，键值映射方式见 _CARD_VIEW
    :param cards: 牌
    :return: 牌的字符串输出形式
    """
    result: str = ''
    for i in cards:
        result += CARD_VIEW[i]

    return result


def _is_consequent(seq, min_len: int) -> bool:
    if seq[-1] >= CARD_2 or len(seq) < min_len:
        return False
    base = seq[0]
    for i in range(0, len(seq)):
        if seq[i] - i != base:
            return False
    return True


ROCKET_BIT = 0
INVALID_BIT = -1
PASS = -2


def _one(di, value) -> int:
    if _is_consequent(di[1], 5):
        return len(di[1]) * 1000 + 100 + value
    return INVALID_BIT


def _two(di, value) -> int:
    if _is_consequent(di[2], 3) and not di[1]:
        return len(di[2]) * 1000 + 200 + value
    return INVALID_BIT


def _three(di, value) -> int:
    # 飞机 或3带1 或3带2
    if _is_consequent(di[3], 1):
        if not di[1]:
            # 无翼
            if not di[2]:
                return len(di[3]) * 1000 + 300 + value

            # 大翼 或3带2
            if len(di[2]) == len(di[3]):
                return 20000 + len(di[3]) * 1000 + 300 + value

        # 小翼 或3带1
        if len(di[1]) + len(di[2]) * 2 == len(di[3]):
            return 10000 + len(di[3]) * 1000 + 300 + value
    return INVALID_BIT


def _four(di, value) -> int:
    if di[3]:
        return -1

    if len(di[4]) == 1:

        # 4带2单
        if len(di[1]) + len(di[2]) * 2 == 2:
            return 11400 + value
        # 4带2双
        elif len(di[2]) == 2 and not di[1]:
            return 21400 + value
    elif len(di[4]) == 2:
        # 无翼航天飞机
        if not di[1] and not di[2]:
            return 2400 + value
        # 带翼航天飞机
        elif len(di[1]) + len(di[2]) * 2 == 4:
            return 12400 + value

    return INVALID_BIT


MAX_COUNT_STRATEGIES: tuple = (_one, _two, _three, _four)


class Combo:
    """
    卡牌组合类
    """

    def __calc_bit_info(self) -> int:

        self._cards.sort()

        # N带0。包括单，对，三带0，炸弹
        if len(np.unique(self._cards)) == 1:
            return 1000 + 100 * len(self._cards) + self._cards[0]

        # 王炸
        if len(self._cards) == 2:
            return ROCKET_BIT if np.sum(self._cards) == CARD_G1 + CARD_G0 else INVALID_BIT

        di, max_count, value = card_to_di(self._cards)

        return MAX_COUNT_STRATEGIES[max_count - 1](di, value)

    def __init__(self):
        self._cards_view: str = ''
        self._cards: np.ndarray = np.array([])
        self._bit_info: int = PASS

    def pass_(self) -> None:
        """
        不出牌。空的手牌
        """
        self._cards_view: str = ''
        self._cards: np.ndarray = np.array([])
        self._bit_info: int = PASS

    def is_valid(self) -> bool:
        """
        本次出牌是否合法
        """
        return self._bit_info != INVALID_BIT

    def is_not_empty(self) -> bool:
        """
        本次出牌是否不为空过
        """
        return self._bit_info >= 0

    def is_rocket(self) -> bool:
        """
        是否为王炸
        """
        return self._bit_info == ROCKET_BIT

    @property
    def cards_view(self) -> str:
        """
        卡牌在控制台上的视图，每张牌之间用空格分开
        """
        return self._cards_view

    @cards_view.setter
    def cards_view(self, v: Union[str, List[str]]):
        if not v:
            self.pass_()
            return

        if type(v) is str:
            v = v.split()
        try:
            self._cards = np.array([VIEW_TO_VALUE[c] for c in v])
        except KeyError:
            self._bit_info = INVALID_BIT
            return

        self._cards.sort()
        self._bit_info = self.__calc_bit_info()
        self._cards_view = cards_view(self._cards)

    @property
    def cards(self) -> np.ndarray:
        """
        卡牌实际值数组
        """
        return self._cards

    @cards.setter
    def cards(self, v):
        if not v:
            self.pass_()
        else:
            self._cards = v
            self._cards.sort()
            self._cards_view = cards_view(self._cards)
            self._bit_info = self.__calc_bit_info()

    @property
    def bit_info(self) -> int:
        """
        bit_info是一个取值范围在[-1, 30000)上的整数。
        卡牌组合非法时bit_info = -1，王炸 bit_info = 0。
        其它情况规则如下：
        bit_info % 100 = 组合用于比较大小的值。组合类型为N带M时，取N中的最大值。其它时候取所有牌最大值。
        bit_info // 100 % 10 = max(牌i的数量)。例如`AAA44`时取3，`667788`时取2
        bit_info // 1000 % 10 = count(max(牌i的数量))。例如`667788`取3，`333444JK`取2
        bin_info // 10000 = N带M中的M。例如`66`取0，`333J`取1。
        @return: bit_info
        """
        return self._bit_info

    @staticmethod
    def is_bomb(bit_info: int) -> bool:
        """
        判断该bit_info是否代表炸弹（不含王炸）
        @param bit_info: 牌的bit info
        @see: Combo
        @return: 是: True; 否: False
        """
        return bit_info // 100 == 14

    @staticmethod
    def bit_value_lt(cur_bit: int, former_bit: int) -> bool:
        return cur_bit % 100 > former_bit % 100

    @staticmethod
    def bit_type_eq(cur_bit: int, former_bit: int) -> bool:
        return cur_bit // 100 == former_bit // 100

    @staticmethod
    def bit_is_valid(cur_bit: int, former_bit: int) -> bool:
        if cur_bit == ROCKET_BIT:
            return True
        if former_bit == ROCKET_BIT:
            return False
        if Combo.is_bomb(cur_bit):
            return not Combo.is_bomb(former_bit) or Combo.bit_value_lt(cur_bit, former_bit)

        return Combo.bit_type_eq(cur_bit, former_bit) and Combo.bit_value_lt(cur_bit, former_bit)

    def __gt__(self, other: Combo):
        return Combo.bit_is_valid(self._bit_info, other._bit_info)

    def __repr__(self):
        return 'Combo: ' + self._cards_view

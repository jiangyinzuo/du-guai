# -*- coding: utf-8 -*-
from typing import Tuple, Dict

from card import *
from card import CARD_VIEW


def has_rocket(card: CardsType) -> bool:
    """
    一副牌中是否含大小王
    """
    return len(card) >= 2 and card[-2] == CARD_G0


def card_lt2(card: CardsType):
    """
    获取一副牌中所有小于2的牌
    """
    idx = np.searchsorted(card, CARD_2)
    return card[:idx]


def card_lt2_two_g(card: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    把一副牌分解成小于2的，等于2的，大小王
    @return: 元组
    """
    return card[card < CARD_2], card[card == CARD_2], card[card > CARD_2]


def card_split(card: CardsType) -> List[np.ndarray]:
    """
    分解不连续的牌
    @note: 2和大小王不应当在里面
    """
    split_li = [i + 1 for i in range(len(card) - 1) if card[i] + 1 < card[i + 1]]
    return np.split(card, split_li)


def card_to_di(card: np.ndarray) -> Tuple[Dict[int, list], int, int]:
    """
    获取统计卡牌数量的字典
    @return: 一个字典，最多牌的数量，最多的牌中的最大值
    """
    di = {1: [], 2: [], 3: [], 4: []}

    count: int = 0
    former_card = card[0]
    for card in card:
        if card == former_card:
            count += 1
        else:
            di[count].append(former_card)
            count = 1
            former_card = card
    di[count].append(former_card)

    max_count: int = 0
    value: int = 0
    for k, v in di.items():
        if v:
            max_count = k
            value = max(v)
    return di, max_count, value


def card_to_suffix_di(card: np.ndarray) -> Tuple[Dict[int, list], int, int]:
    """
    获取统计卡牌后缀数量的字典
    di[1]表示数量大于等于1张的牌面的数组
    """
    di, max_count, value = card_to_di(card)
    di[1].extend(di[2])
    di[1].extend(di[3])
    di[2].extend(di[3])
    di[1].sort()
    di[2].sort()
    return di, max_count, value


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

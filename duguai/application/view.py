# -*- coding: utf-8 -*-
"""
控制台视图相关函数
"""
import numpy as np

from card import CARD_VIEW


def get_cards_view(cards: np.ndarray) -> str:
    """
    获取牌的字符串形式，键值映射方式见 _CARD_VIEW
    :param cards: 牌
    :return: 牌的字符串输出形式
    """
    result: str = ''
    for i in cards:
        result += CARD_VIEW[i]

    return result

# -*- coding: utf-8 -*-
from typing import List

import numpy as np

from card import CARD_G1, CARD_G0, CARD_2
from card.card_helper import card_lt2, card_split, card_to_di


def _get_combo(c_hand: np.ndarray) -> List[List]:
    """
    @param c_hand: 连续的手牌，且不含大小王和2
    """

    if len(c_hand) == 1:
        return [list(c_hand)]

    di, max_count, value = card_to_di(c_hand)

    # 获取所有炸弹
    result: List[List] = [[i, ] * 4 for i in di[4]]

    if len(c_hand) == 2:
        if len(di[3]) == 2:
            return

    return result


def split_card(hand: np.ndarray) -> List:

    if len(hand) == 1:
        return [hand]
    if len(hand) == 2:
        return [hand] if hand[0] == hand[-1] or (hand[0], hand[1]) == (CARD_G0, CARD_G1) else [[hand[0]], hand[1]]

    sp_result = []

    # 添加卡牌2的组合
    card_2_count = np.sum(hand == CARD_2)
    if card_2_count:
        sp_result.append([CARD_2, ] * card_2_count)

    # 添加大小王的组合
    if hand[-1] == CARD_G1:
        if hand[-2] == CARD_G0:
            sp_result.append([CARD_G0, CARD_G1])
        else:
            sp_result.append([CARD_G1])
    elif hand[-1] == CARD_G0:
        sp_result.append([CARD_G0])

    # 添加一般情况
    split_hand = card_split(card_lt2(hand))

    return [_get_combo(i) for i in split_hand] + sp_result

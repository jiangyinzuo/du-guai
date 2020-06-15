# -*- coding: utf-8 -*-

from collections import defaultdict

from card.card_helper import *
from card.card_helper import card_lt_2, card_to_di, card_split

init_q_table = defaultdict(int)


def get_next_state(state: CardType, action: CardType):
    next_state = list(state)
    for card in action:
        next_state.remove(card)
    return next_state


def get_reward(state: CardType, action: list):
    next_state = get_next_state(state, action)
    return 10000 if not next_state else len(action) + decompose_value(next_state) - 1


def get_seq_solo_actions(state: CardType, length: int) -> ActionsType:
    """
    顺子动作
    @param state: 状态
    @param length: 动作长度
    """
    seq_solo = np.unique(state)
    result = []
    for i in range(len(seq_solo) - length + 1):
        result.append(seq_solo[i:i + length])
    return result


def get_single_actions(state: CardType, length: int) -> ActionsType:
    """
    单种牌面的动作（单，对，三，炸弹）
    @param state: 状态
    @param length: 动作长度
    """
    result = []
    last_card = -1
    for i in range(length, len(state) + 1):
        if state[i - 1] == state[i - length] and state[i - 1] != last_card:
            last_card = state[i - 1]
            result.append([last_card] * length)
    return result


ACTIONS_FUNC = {get_single_actions: [1, 2, 3], get_seq_solo_actions: []}
LENGTH_ARGS = []


def get_values(state, actions):
    result = []
    for a in actions:
        result.append(decompose_value(get_next_state(state, a)))
    return result


def _get_good_actions(func, state: CardType, length):
    solo_actions = np.array(func(state, length))
    solo_values = get_values(state, solo_actions)
    args_res = np.argsort(solo_values)
    return args_res[-3:]


def get_good_actions(state: CardType):
    """
    获取当前状态下较好的拆牌
    """
    result = []
    for i in range(1, 4):
        result.extend(_get_good_actions(get_single_actions, state, i))

    for i in range(5, len(np.unique(state))):
        result.extend(_get_good_actions(get_single_actions, state, i))


def decompose_value(card: CardType) -> int:
    """
    获取一副牌的分解值，该函数评估时不考虑N带M
    """
    card = card_lt_2(card)
    di, result, value = card_to_di(card)

    di[1].extend(di[2])
    di[1].extend(di[3])
    di[2].extend(di[3])
    di[1].sort()
    di[2].sort()

    size = 4
    for t in range(1, 4):
        split_card = card_split(di[t])
        size = max(size, max([len(i) for i in split_card]) * t)

    return size if size > 4 and size > result else result

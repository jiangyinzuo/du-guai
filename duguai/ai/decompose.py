# -*- coding: utf-8 -*-

from collections import defaultdict

from card.card_helper import *
from card.card_helper import card_lt_2, card_split

init_q_table = defaultdict(int)


def get_next_state(state: CardType, action: CardType):
    next_state = list(state)
    for card in action:
        next_state.remove(card)
    return next_state


def get_reward(state: CardType, action: list):
    next_state = get_next_state(state, action)
    return 10000 if not next_state else len(action) + decompose_value(next_state) - 1


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


def get_seq_actions(card_list, kind: int, length: int) -> ActionsType:
    result = []
    for i in range(length - 1, len(card_list)):
        if card_list[i] == card_list[i - length + 1] + length - 1:
            result.append(sorted(card_list[i - length + 1: i + 1] * kind))
    return result


def get_values(state, actions) -> np.ndarray:
    result = []
    for a in actions:
        result.append(decompose_value(get_next_state(state, a)))
    return np.array(result)


def _get_good_actions(func, state, **kwargs):
    actions = np.array(
        func(state, kwargs['length']) if 'card_list' not in kwargs.keys() else func(kwargs['card_list'],
                                                                                    kwargs['kind'],
                                                                                    kwargs['length']))
    values = get_values(state, actions) + kwargs['length']
    max_value = np.max(values)

    if 'min_value' in kwargs:
        min_value = kwargs['min_value']
        if max_value < min_value:
            return [], min_value
    args_res = values == max_value
    return actions[args_res].tolist(), max_value


kind_to_min_len = {1: 5, 2: 3, 3: 2, 4: 2}


def get_good_actions(state: CardType):
    """
    获取当前状态下较好的拆牌
    """
    result = []
    max_value = -1
    for i in range(1, 4):
        good_action, max_value = _get_good_actions(get_single_actions, state, min_value=max_value-2, length=i)
        result.extend(good_action)

    di = card_to_suffix_di(state)[0]

    max_value = -1
    for k, min_len in kind_to_min_len.items():
        card_list = di[k]
        for length in range(min_len, len(card_list) + 1):
            good_action, max_value = _get_good_actions(get_seq_actions, state,
                                                       min_value=max_value,
                                                       card_list=card_list,
                                                       kind=k,
                                                       length=length)
            result.extend(good_action)

    return result


def decompose_value(card: CardType) -> int:
    """
    获取一副牌的分解值，该函数评估时不考虑N带M
    """
    card = card_lt_2(card)
    di, result, value = card_to_suffix_di(card)

    size = 4
    for t in range(1, 4):
        split_card = card_split(di[t])
        size = max(size, max([len(i) for i in split_card]) * t)

    return size if size > 4 and size > result else result

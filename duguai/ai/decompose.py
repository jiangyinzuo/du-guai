# -*- coding: utf-8 -*-
"""
斗地主拆牌模块
@author 江胤佐
"""

from card.card_helper import *
from card.card_helper import card_lt2, card_split

"""顺子/连对/飞机/航天飞机最小的长度"""
KIND_TO_MIN_LEN = {1: 5, 2: 3, 3: 2, 4: 2}
MAX_Q = 10000


class Decomposer:
    """
    拆牌类。该类只负责拆出较好的牌组，不考虑其它玩家手牌的情况。
    状态（state）表示目前的手牌。
    动作（action）表示待出的牌。
    Q值：状态-动作的价值，Q(s, a)值越大，则越要在状态s下采取行动a.
        q = d(next_state) + len(a)
    """

    def __init__(self):
        self._state = None
        self._suffix_di: dict = {}

    @staticmethod
    def decompose_value(card: CardsType) -> int:
        """
        获取一副牌的分解值，该函数评估时不考虑N带M
        """
        card = card_lt2(card)
        di, result, value = card_to_suffix_di(card)

        size = 4
        for t in range(1, 4):
            split_card = card_split(di[t])
            size = max(size, max([len(i) for i in split_card]) * t)

        return size if size > 4 and size > result else result

    def _calc_d(self, lt2_state, actions) -> np.ndarray:
        """对每一种状态-动作计算其d值"""
        result = []
        for a in actions:
            next_state = get_next_state(lt2_state, a)
            if next_state:
                result.append(self.decompose_value(next_state))
            else:
                # 该动作打完就没牌了，故d值为最大值
                result.append(MAX_Q)
        return np.array(result)

    def _eval_actions(self, func, lt2_state: CardsType, **kwargs):
        actions = np.array(
            func(lt2_state, kwargs['length']) if 'card_list' not in kwargs.keys() else func(kwargs['card_list'],
                                                                                            kwargs['kind'],
                                                                                            kwargs['length']))

        # q = d(next state) + len(a)
        q_list: np.ndarray = self._calc_d(lt2_state, actions) + kwargs['length']
        max_q = np.max(q_list)

        if 'min_value' in kwargs:
            min_value = kwargs['min_value']
            if max_q < min_value:
                return [], min_value
        args_res = q_list == max_q
        return actions[args_res].tolist(), max_q

    def _get_lt2_good_actions(self, lt2_state: CardsType):
        """
        获取小于2的牌中当前状态下较好的拆牌
        """
        result = []
        max_value = -1
        di, max_count, max_card_value = card_to_suffix_di(lt2_state)
        for i in range(1, max_count + 1):
            good_action, max_value = self._eval_actions(get_single_actions,
                                                        lt2_state,
                                                        min_value=max_value - 2,
                                                        length=i)
            result.extend(good_action)

        max_value = -1
        for k, min_len in KIND_TO_MIN_LEN.items():
            card_list = di[k]
            for length in range(min_len, len(card_list) + 1):
                good_action, max_value = self._eval_actions(get_seq_actions,
                                                            lt2_state,
                                                            min_value=max_value,
                                                            card_list=card_list,
                                                            kind=k,
                                                            length=length)
                result.extend(good_action)

        return result

    def get_good_actions(self, state: CardsType, last_action: CardsType = None) -> List:
        """
        获取较好的行动。
        @param state: 本次牌。若不为空表示跟牌；否则表示出牌
        @param last_action: 上一次出的牌。若不为空表示跟牌；否则表示出牌
        @return: 包含所有好的出牌类型的数组
        """
        self._state = np.array(state)

        # 将手牌分解成不连续的部分
        lt2_cards, eq2_cards, ghosts = card_lt2_two_g(self._state)
        lt2_states = card_split(lt2_cards)
        result = []
        for lt2_state in lt2_states:
            result.extend(self._get_lt2_good_actions(lt2_state))

        return result


def get_next_state(state: CardsType, action: CardsType) -> List[int]:
    """
    获取状态做出动作后的的下一个状态
    @param state: 状态
    @param action: 动作
    @return: 下一个状态
    """
    next_state = list(state)
    for card in action:
        next_state.remove(card)
    return next_state


def get_reward(state: CardsType, action: list) -> int:
    next_state = get_next_state(state, action)
    return 10000 if not next_state else len(action) + Decomposer.decompose_value(next_state) - 1


def get_single_actions(state: CardsType, length: int) -> ActionsType:
    """
    获取所有单种牌面的动作（单，对，三，炸弹）
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
    """
    获取顺子/连对/飞机/炸弹的动作（单，对，三，炸弹）
    """
    result = []
    for i in range(length - 1, len(card_list)):
        if card_list[i] == card_list[i - length + 1] + length - 1:
            result.append(sorted(card_list[i - length + 1: i + 1] * kind))
    return result

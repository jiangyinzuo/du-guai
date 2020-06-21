# -*- coding: utf-8 -*-
"""
斗地主拆牌模块
@author 江胤佐
"""
from __future__ import annotations

from abc import ABCMeta
from typing import Iterable

from card.cards import *
from card.cards import card_lt2, card_split
from card.combo import Combo

"""顺子/连对/飞机/最小的长度"""
KIND_TO_MIN_LEN = {1: 5, 2: 3, 3: 2}
MAX_Q = 10000


class AbstractDecomposer(metaclass=ABCMeta):
    """
    拆牌类。该类只负责拆出较好的牌组，不考虑其它玩家手牌的情况。
    状态（state）表示目前的手牌。
    动作（action）表示待出的牌。
    Q值：状态-动作的价值，Q(s, a)值越大，则越要在状态s下采取行动a.
        q = d(next_state) + len(a)
    """

    @staticmethod
    def decompose_value(card: List[int]) -> int:
        """
        获取一副牌的分解值
        """
        if len(card) == 0:
            return 0
        card = card_lt2(card)
        di, d_value, _ = card_to_suffix_di(card)

        # 顺子/连对
        for t in range(1, 3):
            max_len = max(len(i) for i in card_split(di[t]))
            if max_len >= KIND_TO_MIN_LEN[t]:
                d_value = max(d_value, t * max_len)

        # 3连的个数
        trios = max(len(i) for i in card_split(di[3]))

        # 单的个数, 对子的个数
        solo, pairs = len(di[1]) + len(di[2]) * 2, len(di[2])

        # 3带1，3带2，飞机
        if trios:
            d_value = max(d_value, trios * 3 + (trios if solo >= trios else 0))
            d_value = max(d_value, trios * 3 + (trios * 2 if pairs >= trios else 0))
        # 4带2
        if len(di[4]):
            d_value = max(d_value, 4 + (2 if solo >= 2 or pairs >= 1 else 0))
            d_value = max(d_value, 4 + (4 if pairs >= 2 else 0))

        return d_value

    def _calc_d_values(self, lt2_state: np.ndarray, actions, no_max_q: bool = False) -> np.ndarray:
        """对每一种状态-动作计算其d_value"""
        result = []
        for a in actions:

            # 拆炸弹的惩罚值
            penalty: int = 0
            for card in a:
                if np.sum(lt2_state == card) == 4 and len(a) < 4:
                    penalty = -1
                    break

            next_state: List[int] = get_next_state(lt2_state, a)
            if next_state or no_max_q:
                result.append(self.decompose_value(next_state) + penalty)
            else:
                # 该动作打完就没牌了，故d值为最大值
                result.append(MAX_Q)
        return np.array(result)

    def _eval_actions(self,
                      func,
                      lt2_state: np.ndarray,
                      no_max_q: bool = False,
                      **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        actions = np.array(
            func(lt2_state, kwargs['length']) if 'card_list' not in kwargs.keys() else func(kwargs['card_list'],
                                                                                            kwargs['kind'],
                                                                                            kwargs['length']))
        # q = d(next state) + len(a)
        len_a = kwargs['length'] if 'card_list' not in kwargs.keys() else kwargs['length'] * kwargs['kind']
        q_list: np.ndarray = self._calc_d_values(lt2_state, actions, no_max_q) + len_a
        if len(q_list) == 0:
            return np.array([]), np.array([])
        max_q = np.max(q_list)

        args_res: np.ndarray = q_list == max_q
        return actions[args_res], q_list[args_res]

    def _process_state(self, state: List[int], last_combo: Combo = None):
        self._state = np.array(state)

        # 将手牌分解成不连续的部分
        self._lt2_cards, eq2_cards, self._ghosts = card_lt2_two_g(self._state)
        self._lt2_states: List[np.ndarray] = card_split(self._lt2_cards)
        self.card2_count: int = len(eq2_cards)
        self._output: List[np.ndarray] = []
        self._last_combo: Combo = last_combo

    @staticmethod
    def _max_q_actions(good_actions, q_lists, delta: int = 0, max_q: int = 0) -> np.ndarray:
        if len(q_lists) == 0:
            return np.array([])
        good_actions: np.ndarray = np.array(good_actions)
        return np.array(good_actions[np.array(q_lists) >= max(max_q, np.max(q_lists)) - delta])


class FollowDecomposer(AbstractDecomposer):
    """
    跟牌拆牌器
    """

    def __init__(self):
        self._lt2_di = None

    def _add_bomb(self, bomb_list: list) -> None:
        """添加炸弹"""

        if len(self._ghosts) == 2:
            self._output.append(self._ghosts)

        if self.card2_count == 4:
            self._output.append(np.array([CARD_2] * 4))

        if self._last_combo.is_bomb():
            for card in bomb_list:
                if card > self._last_combo.value:
                    self._output.append(np.array([card, card, card, card]))
        else:
            for card in bomb_list:
                self._output.append(np.array([card, card, card, card]))

    def _add_single(self, kind: int, min_value: int = -1) -> None:

        # 加入ghost
        if kind == 1 and len(self._ghosts) and self._last_combo.value < self._ghosts[-1]:
            self._output.append(np.array([self._ghosts[-1]]))

        # 加入2
        if self.card2_count >= kind and min_value < CARD_2:
            self._output.append(np.array([CARD_2] * kind))

        good_actions: list = []
        q_lists: list = []

        for lt2_state in self._lt2_states:
            good_action, q_list = self._eval_actions(_get_single_actions,
                                                     lt2_state,
                                                     length=kind)
            good_actions.extend(good_action)
            q_lists.extend(q_list)
        if len(good_actions):
            good_actions: np.ndarray = np.array(np.array(good_actions)[np.array(q_lists) == np.max(q_lists)])
            for i in np.unique(good_actions[good_actions > min_value]):
                self._output.append(np.array([i] * kind))

    def _add_seq(self, length: int, kind: int) -> None:
        """
        添加序列（顺子/连对/飞机/航天飞机）
        @param length: 序列长度
        @param kind: 序列种类，顺子取1，连对取2，飞机取3，航天飞机取4
        """
        for lt2_state in self._lt2_states:
            good_actions: list = []
            q_lists: list = []
            max_q = -1
            for k, min_len in KIND_TO_MIN_LEN.items():
                card_list: list = self._lt2_di[k]
                for card_len in range(min_len, len(card_list) + 1):
                    good_action, q_list = self._eval_actions(_get_seq_actions,
                                                             lt2_state,
                                                             True,
                                                             card_list=card_list,
                                                             kind=k,
                                                             length=card_len)
                    if len(q_list) == 0:
                        break
                    if k == kind and length == card_len:
                        good_actions.extend(good_action)
                        q_lists.extend(q_list)
                    max_q = max(max_q, np.max(q_list))
            self._output.extend(self._max_q_actions(good_actions, q_lists, 1, max_q))

    def get_good_follows(self, state: list, last_combo: Combo) -> List[np.ndarray]:
        """
        获取较好的跟牌行动。
        @param state: 当前手牌。
        @param last_combo: 上一次出牌
        @return: 包含所有好的出牌类型的数组
        """
        if last_combo.is_rocket():
            return []
        self._process_state(state, last_combo)

        self._lt2_di = card_to_suffix_di(self._lt2_cards)[0]

        self._add_bomb(self._lt2_di[4])

        if last_combo.has_solo():
            self._add_single(1, self._last_combo.value if last_combo.is_solo() else -1)
        elif last_combo.has_pair():
            self._add_single(2, self._last_combo.value if last_combo.is_pair() else -1)
        if last_combo.is_trio():
            self._add_single(3, self._last_combo.value)
        elif last_combo.is_quartet():
            self._add_single(4, self._last_combo.value)

        if last_combo.is_seq():
            self._add_seq(last_combo.seq_len, last_combo.kind)
        return self._output

    def get_remain_follows_no_take(self, state, last_combo: Combo) -> List[np.ndarray]:
        """
        获取强行拆牌时剩余的跟牌行动。在get_good_follows没有理想结果，但是必须跟牌的情况下。
        @param state: 当前手牌。
        @param last_combo: 上一次出牌
        @note: last combo为N带M时，没有给出M
        @return: 包含剩余不好的出牌类型数组
        """
        if last_combo.is_rocket():
            return []
        self._process_state(state, last_combo)

        self._lt2_di = card_to_suffix_di(self._lt2_cards)[0]

        self._add_bomb(self._lt2_di[4])

        if last_combo.is_single():
            actions = _get_single_actions(state, last_combo.kind)
            self._output.extend((np.array(a) for a in actions if a[-1] > last_combo.value))
        elif last_combo.is_seq():
            actions = _get_seq_actions(state, kind=last_combo.kind, length=last_combo.seq_len)
            self._output.extend((np.array(a) for a in actions if a[-1] > last_combo.value))

        return self._output


class PlayDecomposer(AbstractDecomposer):
    """
    出牌拆牌器
    """

    def _get_lt2_good_actions(self, lt2_state: np.ndarray) -> Iterable:
        """
        获取小于2的牌中当前状态下较好的拆牌
        """
        good_actions: list = []
        q_lists: list = []
        di, max_count, max_card_value = card_to_suffix_di(lt2_state)

        # 拆单、对、三、炸弹
        for kind in range(1, max_count + 1):
            good_action, q_list = self._eval_actions(_get_single_actions,
                                                     lt2_state,
                                                     length=kind)
            good_actions.extend(good_action)
            q_lists.extend(q_list)

        # 拆出顺子、连对、飞机等
        for k, min_len in KIND_TO_MIN_LEN.items():
            card_list = di[k]
            for length in range(min_len, len(card_list) + 1):
                good_action, q_list = self._eval_actions(_get_seq_actions,
                                                         lt2_state,
                                                         card_list=card_list,
                                                         kind=k,
                                                         length=length)
                good_actions.extend(good_action)
                q_lists.extend(q_list)

        return self._max_q_actions(good_actions, q_lists)

    def get_good_plays(self, state: List[int]) -> List[np.ndarray]:
        """
        获取较好的出牌行动。
        @param state: 当前手牌。
        @return: 包含所有好的出牌类型的数组
        """
        self._process_state(state)

        if self.card2_count:
            self._output.append(np.array([CARD_2] * self.card2_count))
        if len(self._ghosts):
            self._output.append(self._ghosts)

        for lt2_state in self._lt2_states:
            self._output.extend(self._get_lt2_good_actions(lt2_state))

        return self._output


def get_next_state(state: np.ndarray, action: np.ndarray) -> List[int]:
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


def _get_single_actions(state: np.ndarray, length: int) -> List[List[int]]:
    """
    获取所有单种牌面的动作（单，对，三，炸弹）
    @param state: 状态
    @param length: 动作长度
    """
    result = []
    last_card = -1
    state = list(state)
    for i in range(length, len(state) + 1):
        if state[i - 1] == state[i - length] and state[i - 1] != last_card and (
                state.count(state[i - 1]) < 4 or length % 2 == 0):
            last_card = state[i - 1]
            result.append([last_card] * length)
    return result


def _get_seq_actions(card_list: list, kind: int, length: int) -> List[List[int]]:
    """
    获取顺子/连对/飞机/炸弹的动作（单，对，三，炸弹）
    """
    result = []
    for i in range(length - 1, len(card_list)):
        if card_list[i] == card_list[i - length + 1] + length - 1:
            result.append(sorted(card_list[i - length + 1: i + 1] * kind))
    return result

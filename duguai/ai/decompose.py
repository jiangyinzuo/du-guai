# -*- coding: utf-8 -*-
"""
斗地主拆牌模块
@author 江胤佐
"""
from __future__ import annotations

import math
from abc import ABCMeta
from collections import defaultdict
from copy import deepcopy
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
    def decompose_value(card: np.ndarray) -> int:
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
            d_value = max(d_value, trios * 5)
        # 4带2
        if len(di[4]):
            d_value = max(d_value, 4 + (2 if solo >= 2 or pairs >= 1 else 0))
            d_value = max(d_value, 4 + (4 if pairs >= 2 else 0))

        return d_value

    def __init__(self):
        # 将要输出的好的牌型列表，每次调用前清空
        self._output: List[np.ndarray] = []

    def _calc_d_values(self, lt2_state: np.ndarray, actions: np.ndarray[np.ndarray], no_max_q: bool = False) \
            -> np.ndarray:
        """对每一种状态-动作计算其d_value"""
        result = []
        for a in actions:

            reward: int = 0
            # 拆炸弹的惩罚值
            for card in a:
                if np.sum(lt2_state == card) == 4 and len(a) < 4:
                    reward = -1
                    break
            else:
                # 3带M加上M的长度（视为 三/飞机 带对子）
                combo = Combo()
                combo.cards = a
                if combo.main_kind == 3:
                    reward = combo.seq_len * 2

            next_state: np.ndarray = get_next_state(lt2_state, a)
            if next_state.size > 0 or no_max_q:
                result.append(self.decompose_value(next_state) + reward)
            else:
                # 该动作打完就没牌了，故d值为最大值
                result.append(MAX_Q)
        return np.array(result)

    def _eval_actions(self,
                      func,
                      lt2_state: np.ndarray,
                      all_actions: bool,
                      no_max_q: bool = False,
                      **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        actions = np.array(
            func(lt2_state, kwargs['length']) if 'card_list' not in kwargs.keys() else func(kwargs['card_list'],
                                                                                            kwargs['kind'],
                                                                                            kwargs['length']))
        # q = d(next state) + len(a)
        len_a = kwargs['length'] if 'card_list' not in kwargs.keys() else kwargs['length'] * kwargs['kind']

        # 计算lt2_state下每一个action的q值
        q_list: np.ndarray = self._calc_d_values(lt2_state, actions, no_max_q) + len_a
        if len(q_list) == 0:
            return np.array([]), np.array([])

        # 若需要所有动作及Q值，则直接返回
        if all_actions:
            return actions, q_list
        else:
            max_q = np.max(q_list)
            args_res: np.ndarray = q_list == max_q
            return actions[args_res], q_list[args_res]

    def _process_state(self, state: np.ndarray, last_combo: Combo = None):
        self._state = np.array(state)

        # 将手牌分解成不连续的部分
        self._lt2_cards, eq2_cards, self._ghosts = card_lt2_two_g(self._state)
        self._lt2_states: List[np.ndarray] = card_split(self._lt2_cards)
        self.card2_count: int = len(eq2_cards)

        # 方法将输出的列表，每次调用方法前清空列表
        self._output.clear()

        self._last_combo: Combo = last_combo

    @staticmethod
    def _max_q_actions(good_actions, q_lists, delta: int = 0, max_q: int = 0) -> np.ndarray:
        if len(q_lists) == 0:
            return np.array([])
        good_actions: np.ndarray = np.array(good_actions)
        return np.array(good_actions[np.array(q_lists) >= max(max_q, np.max(q_lists)) - delta])

    def _get_good_actions_and_q_lists(self, lt2_state: np.ndarray, all_actions: bool) \
            -> Tuple[List[np.ndarray], List[int]]:
        """获取一个lt2_state下所有的actions及其对应的q_lists"""

        good_actions: list = []
        q_lists: list = []
        di, max_count, max_card_value = card_to_suffix_di(lt2_state)

        # 拆单、对、三、炸弹
        for kind in range(1, max_count + 1):
            good_action, q_list = self._eval_actions(_get_single_actions,
                                                     lt2_state,
                                                     all_actions,
                                                     length=kind)
            good_actions.extend(good_action)
            q_lists.extend(q_list)

        # 拆出顺子、连对、飞机等
        for k, min_len in KIND_TO_MIN_LEN.items():
            card_list = di[k]
            for length in range(min_len, len(card_list) + 1):
                good_action, q_list = self._eval_actions(_get_seq_actions,
                                                         lt2_state,
                                                         all_actions,
                                                         card_list=card_list,
                                                         kind=k,
                                                         length=length)
                good_actions.extend(good_action)
                q_lists.extend(q_list)
        return good_actions, q_lists


class FollowDecomposer(AbstractDecomposer):
    """
    跟牌拆牌器
    """

    def __init__(self):
        super().__init__()

        # 存放带牌的列表
        self._take_lists: Dict[int, List[np.ndarray]]

        # 存放主牌的列表
        self._main_lists: Dict[int, List[np.ndarray]]

        # 存放主牌+带牌的列表
        self._main_take_lists: Dict[int, List[np.ndarray]]

        # 炸弹列表
        self._bomb_list: List[np.ndarray]

        # 最大的出牌
        self._max_main_takes: np.ndarray

    def _init(self):
        """初始化，key代表max_q - q，key越小拆得越好，越要优先选择"""
        self._take_lists: Dict[int, List[np.ndarray]] = defaultdict(list)
        self._main_lists: Dict[int, List[np.ndarray]] = defaultdict(list)
        self._max_main_takes = self._last_combo.cards

    def _add_bomb(self, bomb_list: list) -> None:
        """添加炸弹"""

        self._bomb_list = []

        # 添加王炸
        if len(self._ghosts) == 2:
            self._bomb_list.append(self._ghosts)

        # 添加4个2炸弹
        if self.card2_count == 4:
            self._bomb_list.append(np.array([CARD_2] * 4))

        if self._last_combo.is_bomb():
            for card in bomb_list:
                if card > self._last_combo.value:
                    self._bomb_list.append(np.array([card, card, card, card]))
        else:
            for card in bomb_list:
                self._bomb_list.append(np.array([card, card, card, card]))

    def _thieve_valid_ghost(self):
        """加入单只王"""

        if self._ghosts.size:
            if self._last_combo.is_solo() \
                    and self._last_combo.main_kind == 1 and self._last_combo.value < self._ghosts[-1]:
                self._main_lists[2].append(self._ghosts[-1:])
            elif self._last_combo.take_kind == 1:
                self._take_lists[2].append(self._ghosts[-1:])

    def _thieve_valid_card2(self):
        if self.card2_count:
            if self._last_combo.is_single() \
                    and self._last_combo.main_kind <= self.card2_count and self._last_combo.value < CARD_2:
                self._main_lists[0].append(np.array([CARD_2] * self._last_combo.main_kind))

            if self._last_combo.take_kind <= self.card2_count:
                self._take_lists[2].append(np.array([CARD_2] * self._last_combo.take_kind))

    def _add_takes(self, main_q: int, main_seq: np.ndarray, take_count: int) -> Tuple[int, np.ndarray]:
        tk = 0
        main_takes: np.ndarray = np.array(main_seq)
        total_delta_q = main_q
        for delta_q, take_list in self._take_lists.items():
            for take in take_list:
                if take[0] not in main_seq:
                    tk += 1
                    total_delta_q += delta_q
                    main_takes = np.concatenate((main_takes, take))
                if tk == take_count:
                    return total_delta_q, main_takes
        return 0, np.array([])

    def _thieve_valid_main_takes(self):
        take_count = self._last_combo.seq_len

        # 炸弹是4带2
        if self._last_combo.main_kind == 4:
            take_count *= 2

        self._main_take_lists = defaultdict(list)

        if self._main_lists:
            main_q = min(self._main_lists.keys())
            for main_seq in self._main_lists[main_q]:
                total_delta_q, main_takes = self._add_takes(main_q, main_seq, take_count)
                if main_takes.size > 0:
                    self._main_take_lists[total_delta_q].append(main_takes)

            self._max_main_takes = self._add_takes(0, self._max_combo.cards, take_count)[1]

    @classmethod
    def _delta_q(cls, _max_q, _q):
        return (_max_q - _q) if _max_q - _q < 1000 else (_max_q - MAX_Q - _q)

    def _add_to_main_lists_and_find_max(self, a: np.ndarray, q: int, max_q: int):

        main_kind = self._last_combo.main_kind
        seq_len = self._last_combo.seq_len
        value = self._last_combo.value

        combo = Combo()
        combo.cards = a
        # 筛选符合规则的主牌
        if combo.value > value and combo.main_kind == main_kind and combo.seq_len == seq_len:
            self._main_lists[self._delta_q(max_q, q)].append(a)
            # 仅对比主牌大小，不关心是否带了牌
            if combo.value > self._max_combo.value:
                self._max_combo = deepcopy(combo)

    def _best_main_takes(self):
        self._thieve_valid_main_takes()
        if not self._main_take_lists:
            return 0, []
        min_delta_q = min(self._main_take_lists.keys())
        return min_delta_q, self._main_take_lists[min_delta_q]

    def _thieve_valid_actions(self) -> Tuple[int, List[np.ndarray]]:

        take_kind = self._last_combo.take_kind

        self._thieve_valid_ghost()
        self._thieve_valid_card2()

        self._max_combo: Combo = self._last_combo

        for lt2_state in self._lt2_states:
            if lt2_state.size > 0:
                actions, q_lists = super(FollowDecomposer, self)._get_good_actions_and_q_lists(lt2_state, True)
                max_q: int = max(q_lists)
                for a, q in zip(actions, q_lists):

                    # 对子可以视为2个单加入take列表
                    if len(a) == take_kind or len(a) == 2 and take_kind == 1:
                        self._take_lists[self._delta_q(max_q, q)].append(a[:take_kind])

                    self._add_to_main_lists_and_find_max(a, q, max_q)

        if not self._main_lists:
            return 0, []

        if take_kind:
            return self._best_main_takes()

        self._max_main_takes = self._max_combo.cards
        min_delta_q = min(self._main_lists.keys())
        return min_delta_q, self._main_lists[min_delta_q]

    def get_good_follows(self, state: np.ndarray, last_combo: Combo) \
            -> Tuple[List[np.ndarray], int, List[np.ndarray], np.ndarray]:
        """
        尽量给出较好的跟牌行动。
        @param state: 当前手牌。
        @param last_combo: 上一次出牌
        @return: 四元组：炸弹, 最好的组合 - 最好的跟牌(数字越大越不应该这样拆牌), 好的出牌的数组, 最大的出牌
        """
        if last_combo.is_rocket():
            return [], 0, [], np.array([], dtype=int)

        self._process_state(state, last_combo)
        self._init()

        min_delta_q, self._output = self._thieve_valid_actions()

        self._add_bomb(card_to_di(self._lt2_cards)[0][4])

        self._max_combo.cards = self._max_main_takes

        return self._bomb_list, min_delta_q, self._output, (
            self._max_main_takes if self._max_combo > last_combo else np.array([], dtype=int))


class PlayHand:
    """
    出牌时，根据d_actions对手牌进行进一步分类
    """

    def __init__(self, cards: np.ndarray, d_actions: List[np.ndarray]):
        """
        初始化Hand类
        @param cards: 玩家的初始手牌
        @param d_actions: 经过初步分解后的行动。
        @see PlayDecomposer
        """
        self._cards: np.ndarray = cards

        self._solos: List[np.ndarray] = []
        self._pairs: List[np.ndarray] = []

        self._trios: List[np.ndarray] = []
        self._bombs: List[np.ndarray] = []
        self._planes: List[np.ndarray] = []

        self._trios_take: List[np.ndarray] = []
        self._planes_take: List[np.ndarray] = []
        self._bombs_take: List[np.ndarray] = []

        self._seq_solo5: List[np.ndarray] = []
        self._other_seq: List[np.ndarray] = []
        self._has_rocket: bool = False

        self._min_solo: int = np.min(cards)
        self._max_solo: int = np.max(cards)

        self.__part_action(d_actions)

        self.merge_main_takes(self._planes, self._planes_take)
        self.merge_main_takes(self._trios, self._trios_take)
        self.merge_main_takes(self._bombs, self._bombs_take)

    def __part_action(self, d_actions: List[np.ndarray]) -> None:
        """对d_action进行初步分类"""

        for a in d_actions:

            # 含王炸
            if len(a) == 2 and a[-1] == CARD_G1:
                self._has_rocket = True

            # 分类单牌
            elif 1 <= len(a) <= 4:
                (self.solos, self.pairs, self.trios, self.bombs)[len(a) - 1].append(a)

            # 分类长度为5的单顺
            elif len(a) == 5 and a[0] == a[1] - 1 == a[3] - 3 == a[4] - 4:
                self._seq_solo5.append(a)

            # 分类飞机
            elif len(a) % 3 == 0 and a[0] == a[2] and a[0] != a[3]:
                self._planes.append(a)

            # 分类其它顺子
            else:
                self.other_seq.append(a)

    @staticmethod
    def _choose_takes(take_list: List[np.ndarray], main_part: np.ndarray, take_count: int, split_pair: bool = False):

        main_part = np.concatenate([main_part] + take_list[:take_count])
        if split_pair:
            main_part = np.concatenate([main_part, take_list[take_count][:1]])

        return main_part

    def merge_main_takes(self, main_list: List[np.ndarray], extended_target: List[np.ndarray]):
        """
        合并主要部分与带的牌
        """
        main_take_list: List[np.ndarray] = []
        for main_part in main_list:

            # 防止main part带上自己的部分，例如 7 7 7不能带7
            temp_solos: List[np.ndarray] = [i for i in self._solos if i[0] not in np.unique(main_part)]
            temp_pairs: List[np.ndarray] = [i for i in self._pairs if i[0] not in np.unique(main_part)]

            take_count: int = math.ceil(main_part.size / 3)
            if len(temp_solos) >= take_count and len(temp_pairs) >= take_count:
                if np.mean(temp_solos) > np.mean(temp_pairs):
                    main_take_list.append(self._choose_takes(temp_solos, main_part, take_count))
                else:
                    main_take_list.append(self._choose_takes(temp_pairs, main_part, take_count))
            elif len(temp_pairs) >= take_count:
                main_take_list.append(self._choose_takes(temp_pairs, main_part, take_count))
            elif len(temp_solos) >= take_count:
                main_take_list.append(self._choose_takes(temp_solos, main_part, take_count))
            elif len(temp_solos) + 2 * len(temp_pairs) >= take_count:
                len_solos = len(temp_solos)
                main_part = self._choose_takes(temp_solos, main_part, len_solos)
                main_take_list.append(
                    self._choose_takes(
                        temp_pairs, main_part, (take_count - len_solos) // 2, (take_count - len_solos) % 2 == 1
                    )
                )
            else:
                main_take_list.append(main_part)
        extended_target.extend(main_take_list)

    @property
    def cards(self) -> np.ndarray:
        """玩家原始的卡牌"""
        return self._cards

    @property
    def solos(self) -> List[np.ndarray]:
        """单"""
        return self._solos

    @property
    def pairs(self) -> List[np.ndarray]:
        """对"""
        return self._pairs

    @property
    def trios(self) -> List[np.ndarray]:
        """三"""
        return self._trios

    @property
    def trios_take(self):
        """三带M"""
        return self._trios_take

    @property
    def bombs(self) -> List[np.ndarray]:
        """炸弹"""
        return self._bombs

    @property
    def bombs_take(self) -> List[np.ndarray]:
        """四带2"""
        return self._bombs_take

    @property
    def planes(self) -> List[np.ndarray]:
        """飞机(不带M)"""
        return self._planes

    @property
    def planes_take(self):
        """飞机(带M)"""
        return self._planes_take

    @property
    def other_seq(self) -> List[np.ndarray]:
        """其它各种序列"""
        return self._other_seq

    @property
    def seq_solo5(self) -> List[np.ndarray]:
        """长度为5的单顺"""
        return self._seq_solo5

    @property
    def has_rocket(self) -> bool:
        """是否有王炸"""
        return self._has_rocket

    @property
    def min_solo(self) -> int:
        """强拆的最小单牌"""
        return self._min_solo

    @property
    def max_solo(self) -> int:
        """强拆的最大单牌"""
        return self._max_solo

    def __repr__(self):
        return 'PlayHand: ' + repr(self.__dict__)


class PlayDecomposer(AbstractDecomposer):
    """
    出牌拆牌器
    """

    def _get_lt2_good_actions(self, lt2_state: np.ndarray) -> Iterable:
        """
        获取小于2的牌中当前状态下较好的拆牌
        """
        actions, q_lists = super(PlayDecomposer, self)._get_good_actions_and_q_lists(lt2_state, False)

        return self._max_q_actions(actions, q_lists)

    def get_good_plays(self, cards: np.ndarray) -> PlayHand:
        """
        获取较好的出牌行动。
        @param cards: 当前手牌。
        @return: 包含所有好的出牌类型的数组
        """
        self._process_state(cards)

        if self.card2_count:
            self._output.append(np.array([CARD_2] * self.card2_count))
        if self._ghosts.size > 0:
            self._output.append(self._ghosts)

        for lt2_state in self._lt2_states:
            self._output.extend(self._get_lt2_good_actions(lt2_state))

        return PlayHand(cards, self._output)


def get_next_state(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """
    获取状态做出动作后的的下一个状态
    @param state: 状态
    @param action: 动作
    @return: 下一个状态
    """
    next_state = list(state)
    for card in action:
        next_state.remove(card)
    return np.array(next_state)


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

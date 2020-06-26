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
from functools import cmp_to_key
from typing import Optional

from duguai.card.cards import *
from duguai.card.cards import card_lt2, card_split
from duguai.card.combo import Combo

"""顺子/连对/最小的长度"""
KIND_TO_MIN_LEN = {1: 5, 2: 3}
MAX_Q = 10000


def _most_value(x):
    return np.argmax(np.bincount(x))


MAX_VALUE_CMP = cmp_to_key(lambda x, y: max(x) - max(y))
MOST_VALUE_CMP = cmp_to_key(lambda x, y: _most_value(x) - _most_value(y))


class AbstractDecomposer(metaclass=ABCMeta):
    """
    拆牌类。该类只负责拆出较好的牌组，不考虑其它玩家手牌的情况。
    状态（state）表示目前的手牌。
    动作（action）表示待出的牌。
    Q值：状态-动作的价值，Q(s, a)值越大，则越要在状态s下采取行动a.
        q = d(next_state) + len(a)
    """

    @classmethod
    def decompose_value(cls, card_after: np.ndarray) -> int:
        """
        获取一副牌的分解值
        """
        if len(card_after) == 0:
            return 0
        card_after = card_lt2(card_after)
        di, d_value, _ = card_to_suffix_di(card_after)

        # 顺子/连对
        for t in range(1, 3):
            max_len = max(len(i) for i in card_split(di[t]))
            if max_len >= KIND_TO_MIN_LEN[t]:
                d_value = max(d_value, t * max_len)

        return d_value

    @classmethod
    def _delta_q(cls, _max_q, _q):
        return (_max_q - _q) if _max_q - _q < 1000 else (_max_q - MAX_Q + 1 - _q)

    def _calc_q(self, lt2_state: np.ndarray, actions: np.ndarray[np.ndarray]) -> np.ndarray:
        """对每一种状态-动作计算其Q"""
        result = []
        for a in actions:

            reward: int = 0
            # 拆炸弹的惩罚值，保证在 5 5 5 5 6的情况下拆出炸弹而非三带一
            for card in a:
                if np.sum(lt2_state == card) == 4 and len(a) < 4:
                    reward = -1
                    break

            next_state: np.ndarray = get_next_state(lt2_state, a)
            if next_state.size > 0:
                d_value = self.decompose_value(next_state)
                result.append(d_value + reward + len(a))
            else:
                # 该动作打完就没牌了，故d值为最大值
                result.append(MAX_Q + reward + len(a))
        return np.array(result)

    def _eval_actions(self,
                      func,
                      lt2_state: np.ndarray,
                      **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        actions = np.array(
            func(lt2_state, kwargs['length']) if 'card_list' not in kwargs.keys() else func(kwargs['card_list'],
                                                                                            kwargs['kind'],
                                                                                            kwargs['length']))
        # q = d(next state) + len(a)
        # 计算lt2_state下每一个action的q值
        q_list: np.ndarray = self._calc_q(lt2_state, actions)
        if len(q_list) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        return actions, q_list

    def _process_card(self, card: np.ndarray):

        # 将手牌分解成不连续的部分
        self._lt2_cards, eq2_cards, self._ghosts = card_lt2_two_g(card)
        self._lt2_states: List[np.ndarray] = card_split(self._lt2_cards)
        self.card2_count: int = len(eq2_cards)

    def _get_all_actions_and_q_lists(self, lt2_state: np.ndarray) -> int:
        """获取一个lt2_state下所有的actions及其对应的q_lists"""

        di, max_count, max_card_value = card_to_suffix_di(lt2_state)

        # solo pair trio bomb plane other
        self._actions = [[], [], [], [], [], []]
        self._q_lists = [np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int),
                         np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)]

        # solo pair trio bomb
        for i in range(1, 5):
            self._actions[i - 1], self._q_lists[i - 1] = self._eval_actions(_get_single_actions, lt2_state, length=i)

        # plane
        for length in range(3, len(di[3]) + 1):
            seq_actions, seq_q_list = self._eval_actions(_get_seq_actions,
                                                         lt2_state,
                                                         card_list=di[3],
                                                         kind=3,
                                                         length=length)
            self._actions[4].extend(seq_actions)
            self._q_lists[4] = np.concatenate([self._q_lists[4], seq_q_list])

        # 拆出顺子、连对
        for k, min_len in KIND_TO_MIN_LEN.items():
            card_list = di[k]
            for length in range(min_len, len(card_list) + 1):
                seq_actions, seq_q_list = self._eval_actions(_get_seq_actions,
                                                             lt2_state,
                                                             card_list=card_list,
                                                             kind=k,
                                                             length=length)
                self._actions[5].extend(seq_actions)
                self._q_lists[5] = np.concatenate([self._q_lists[5], seq_q_list])

        max_q = 0
        for q_list in self._q_lists:
            if q_list.size:
                max_q = max(np.max(q_list), max_q)
        return max_q


class FollowDecomposer(AbstractDecomposer):
    """
    跟牌拆牌器
    """

    def __init__(self):
        self._output: Optional[List[np.ndarray]] = None

        # 存放带牌的列表
        self._take_lists: Optional[Dict[int, List[np.ndarray]]] = None

        # 存放主牌的列表
        self._main_lists: Optional[Dict[int, List[np.ndarray]]] = None

        # 存放主牌+带牌的列表
        self._main_take_lists: Optional[Dict[int, List[np.ndarray]]] = None

        # 炸弹列表
        self._bomb_list: Optional[List[np.ndarray]] = None

        # 仅维护主牌大小
        self._max_combo: Optional[Combo] = None

        self._last_combo: Optional[Combo] = None

        self._main_kind: Optional[int] = None
        self._take_kind: Optional[int] = None

    def _add_bomb(self, bomb_list: list) -> None:
        """添加炸弹"""

        self._bomb_list: List[np.ndarray] = []

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

    def _add_valid_ghost(self):
        """加入单只王。在此之前先加入2"""

        if self._ghosts.size:
            if self._last_combo.is_solo() \
                    and self._last_combo.main_kind == 1 and self._last_combo.value < self._ghosts[-1]:
                self._main_lists[2].append(self._ghosts[-1:])
                self._max_combo.cards = self._ghosts[-1:]
            elif self._max_combo.take_kind == 1:
                self._take_lists[2].append(self._ghosts[-1:])

    def _add_valid_card2(self):
        """加入合法的2，之后再加入王"""
        if self.card2_count:
            if self._last_combo.is_single() \
                    and self._last_combo.main_kind <= self.card2_count and self._last_combo.value < CARD_2:
                self._main_lists[self._max_combo.main_kind].append(np.array([CARD_2] * self._max_combo.main_kind))
                self._max_combo.cards = [CARD_2] * self._max_combo.main_kind
            if self._last_combo.take_kind <= self.card2_count:
                # 2的价值比正常牌+1
                self._take_lists[self._last_combo.take_kind + 1].append(np.array([CARD_2] * self._last_combo.take_kind))

    def __merge_takes_to_main_seq(self, main_q: int, main_seq: np.ndarray, take_count: int) -> Tuple[int, np.ndarray]:
        tk = 0
        main_takes: np.ndarray = np.array(main_seq)
        total_delta_q = main_q

        # 从小到大遍历_take_lists，保证先合并最佳takes
        for delta_q, take_list in sorted(self._take_lists.items()):
            for take in take_list:
                if take[0] not in main_seq:
                    tk += 1
                    total_delta_q += delta_q
                    main_takes = np.concatenate((main_takes, take))
                if tk == take_count:
                    return total_delta_q, main_takes
        return 0, np.array([])

    def _merge_valid_main_takes(self) -> None:
        """将合法的主牌和带牌拼接起来"""

        # 非炸弹是3带1单/1对，炸弹是4带2
        take_count = self._last_combo.seq_len
        if self._last_combo.main_kind == 4:
            take_count *= 2

        self._main_take_lists = defaultdict(list)

        for take_list in self._take_lists.values():
            take_list.sort(key=MAX_VALUE_CMP)

        if self._main_lists:

            # 挑选最佳的main_list，并排序
            main_q = min(self._main_lists.keys())
            self._main_lists[main_q].sort(key=MAX_VALUE_CMP)

            for main_seq in self._main_lists[main_q]:
                total_delta_q, main_takes = self.__merge_takes_to_main_seq(main_q, main_seq, take_count)
                if main_takes.size > 0:
                    # 将得到的main_takes根据价值好坏加入相应的列表中
                    self._main_take_lists[total_delta_q].append(main_takes)

            # 得到最大的main_takes
            self._max_main_takes = self.__merge_takes_to_main_seq(0, self._max_combo.cards, take_count)[1]

    def _update_main_lists_and_find_max(self, a: np.ndarray, q: int, max_q: int) -> None:
        """在action有效的情况下加入到主列表，并更新最大值"""
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
        if not self._main_take_lists:
            return 0, []
        min_delta_q = min(self._main_take_lists.keys())
        self._main_take_lists[min_delta_q].sort(key=MOST_VALUE_CMP)
        return min_delta_q, self._main_take_lists[min_delta_q]

    def _append_takes(self, length: int, kind: int, max_q):
        for a, q in zip(self._actions[kind - 1], self._q_lists[kind - 1]):
            self._take_lists[self._delta_q(max_q, q)].append(a[:length])

    def _add_valid_lt2_actions(self):
        for lt2_state in self._lt2_states:
            if lt2_state.size > 0:
                max_q: int = super(FollowDecomposer, self)._get_all_actions_and_q_lists(lt2_state)

                # 把单或者对加入_take_lists，对子可以视为2个单加入take列表
                if self._take_kind == 1:
                    self._append_takes(1, 1, max_q)
                    self._append_takes(1, 2, max_q)
                elif self._take_kind == 2:
                    self._append_takes(2, 2, max_q)

                for actions, q_list in zip(self._actions, self._q_lists):
                    for a, q in zip(actions, q_list):
                        # 将合法的action加入到_main_lists，同时更新最大的main_kind
                        self._update_main_lists_and_find_max(a, q, max_q)

    def _thieve_valid_actions(self) -> Tuple[int, List[np.ndarray]]:
        """根据last combo的限制，筛选出有效且较好的动作"""

        self._add_valid_card2()
        self._add_valid_ghost()
        self._add_valid_lt2_actions()

        if not self._main_lists:
            return 0, []

        if self._take_kind:
            self._merge_valid_main_takes()
            return self._best_main_takes()
        else:
            self._max_main_takes = self._max_combo.cards
            min_delta_q = min(self._main_lists.keys())

            self._main_lists[min_delta_q].sort(key=MAX_VALUE_CMP)
            return min_delta_q, self._main_lists[min_delta_q]

    def _init(self, last_combo: Combo):
        # 初始化，key代表max_q - q，key越小拆得越好，越要优先选择
        self._take_lists: Dict[int, List[np.ndarray]] = defaultdict(list)
        self._main_lists: Dict[int, List[np.ndarray]] = defaultdict(list)
        self._output = []

        # max_combo仅保留主要部分，忽略带的部分
        self._max_combo = deepcopy(last_combo)
        self._max_main_takes = self._max_combo.cards
        self._last_combo = last_combo

        self._main_kind = self._max_combo.main_kind
        self._take_kind = self._max_combo.take_kind

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

        self._process_card(state)
        self._init(last_combo)

        min_delta_q, self._output = self._thieve_valid_actions()

        self._add_bomb(card_to_di(self._lt2_cards)[0][4])

        self._max_combo.cards = self._max_main_takes

        return self._bomb_list, min_delta_q, self._output, (
            self._max_main_takes if self._max_combo > last_combo else np.array([], dtype=int))


class PlayHand:
    """
    出牌时，根据d_actions对手牌进行进一步分类
    """

    def __init__(self, min_solo: int, max_solo: int):
        """
        初始化Hand类
        @see PlayDecomposer
        """
        # solo pair trio bomb
        self._singles: List[List[np.ndarray]] = [[], [], [], []]

        self._planes: List[np.ndarray] = []

        self._trios_take: List[np.ndarray] = []
        self._planes_take: List[np.ndarray] = []
        self._bombs_take: List[np.ndarray] = []

        self._seq_solo5: List[np.ndarray] = []
        self._other_seq: List[np.ndarray] = []
        self._has_rocket: bool = False

        self._min_solo: int = min_solo
        self._max_solo: int = max_solo

    def add_to_hand(self, card_lists: List[Dict[int, List[np.ndarray]]]):
        """将各种类型牌加入到PlayHand中"""
        for i in range(4):
            if card_lists[i].keys():
                min_delta_q = min(card_lists[i].keys())
                self._singles[i] = card_lists[i][min_delta_q]
                self._singles[i].sort(key=MAX_VALUE_CMP)

        # plane
        if card_lists[4].keys():
            min_delta_q = min(card_lists[4].keys())
            self._planes = card_lists[4][min_delta_q]
            self._planes.sort(key=MAX_VALUE_CMP)

        if card_lists[5].keys():
            min_delta_q = min(card_lists[5].keys())
            for action in card_lists[5][min_delta_q]:
                if action.size == 5:
                    self._seq_solo5.append(action)
                else:
                    self._other_seq.append(action)
            self._seq_solo5.sort(key=MAX_VALUE_CMP)

        self._merge_main_takes(self._planes, self._planes_take)
        self._merge_main_takes(self._singles[2], self._trios_take)
        self._merge_main_takes(self._singles[3], self._bombs_take)

    @staticmethod
    def _choose_takes(take_list: List[np.ndarray], main_part: np.ndarray, take_count: int, split_pair: bool = False):

        main_part = np.concatenate([main_part] + take_list[:take_count])
        if split_pair:
            main_part = np.concatenate([main_part, take_list[take_count][:1]])

        return main_part

    def _merge_main_takes(self, main_list: List[np.ndarray], extended_target: List[np.ndarray]):
        """
        合并主要部分与带的牌
        """
        main_take_list: List[np.ndarray] = []
        for main_part in main_list:

            # 防止main part带上自己的部分，例如 7 7 7不能带7
            temp_pairs: List[np.ndarray] = [i for i in self._singles[1] if i[0] not in np.unique(main_part)]
            temp_solos: List[np.ndarray] = [i for i in self._singles[0] if
                                            i[0] not in np.unique(main_part) and i[0] not in np.unique(temp_pairs)]

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
        extended_target.sort(key=MOST_VALUE_CMP)

    @property
    def solos(self) -> List[np.ndarray]:
        """单"""
        return self._singles[0]

    @property
    def pairs(self) -> List[np.ndarray]:
        """对"""
        return self._singles[1]

    @property
    def trios(self) -> List[np.ndarray]:
        """三"""
        return self._singles[2]

    @property
    def trios_take(self):
        """三带M"""
        return self._trios_take

    @property
    def bombs(self) -> List[np.ndarray]:
        """炸弹"""
        return self._singles[3]

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
    基于贪心法的斗地主出牌时的拆牌算法。
    出牌时仅考虑强行拆最大和最小的单牌。其余牌型均按照最佳拆牌给出。

    定义c为一张牌，由斗地主规则可知，c ∈ [1, 15] ∩ Z+。
    定义s表示当前玩家拥有的所有牌的序列，s = (c1, c2, ..., ci)。
    定义a为一次符合斗地主规则的出牌的序列，a = (c1, c2, ..., ci)。

    记s下满足规则的所有拆牌动作的集合为A_s，a∈A_s。

    用函数D(a)来计算a拆牌的好坏。D(a)定义如下：
                D(a) = len(a) + max( max(len(a')) , 1) - 拆炸弹的数量

    其中定义域a∈A，值域D(a)∈Z+, a' ∈ s - a
    D(a)越大，拆牌越合理。

    算法如下:
    1. 将s分成连续的若干段、二和大小王，例如(1,1,2,2,5,5,7,10,13,14)分成(1,1,2,2) (5,5,) (7) (10) (13) (14)
    2. 将大小王和二加入到最佳拆牌序列A‘中
    3. 对每一段序列si，计算不带牌的动作a的 D(a)
    4. 合并主牌和带牌的D(a)
    5. 输出argmax(D(a))
    """

    def __init__(self):
        self.cards_q_maps_list: Optional[List[Dict[int, List[np.ndarray]]]] = None

    def _map_actions(self, actions, q_list, max_q: int, idx: int):
        for a, q in zip(actions, q_list):
            if max_q == q:
                self.cards_q_maps_list[idx][max_q - q].append(a)

    def get_good_plays(self, cards: np.ndarray) -> PlayHand:
        """
        获取较好的出牌行动。
        @param cards: 当前手牌。
        @return: 包含所有好的出牌类型的数组
        """
        self._process_card(cards)
        self.cards_q_maps_list = [defaultdict(list), defaultdict(list),
                                  defaultdict(list), defaultdict(list),
                                  defaultdict(list), defaultdict(list)]

        play_hand = PlayHand(np.min(cards), np.max(cards))

        for lt2_state in self._lt2_states:
            if lt2_state.size > 0:
                max_q = self._get_all_actions_and_q_lists(lt2_state)
                i = 0
                for actions, q_list in zip(self._actions, self._q_lists):
                    self._map_actions(actions, q_list, max_q, i)
                    i += 1

        if self.cards_q_maps_list[0].keys() and self.cards_q_maps_list[1].keys():
            min_key = min(self.cards_q_maps_list[0].keys())
            min_key2 = min(self.cards_q_maps_list[1].keys())
            self.cards_q_maps_list[0][min_key] = [i for i in self.cards_q_maps_list[0][min_key] if
                                                  i[0] not in np.unique(self.cards_q_maps_list[1][min_key2])]
        if self.card2_count:
            self.cards_q_maps_list[self.card2_count - 1][0].append(np.array([CARD_2] * self.card2_count))

        if self._ghosts.size == 2:
            play_hand._has_rocket = True
        elif self._ghosts.size == 1:
            self.cards_q_maps_list[0][0].append(self._ghosts)

        play_hand.add_to_hand(self.cards_q_maps_list)
        return play_hand


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

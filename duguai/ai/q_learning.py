# -*- coding: utf-8 -*-
"""
Q-Learning算法相关模块
该模块包含3个智能体（Agent），分别执行随机策略、查询Q表（不训练）、Q-Learning

@author: 江胤佐
"""
import logging
import os
from abc import ABCMeta, abstractmethod, ABC
from random import sample, random
from typing import List, Union, Optional, Tuple

import numpy as np

from duguai import mode
from duguai.ai.provider import FollowProvider, PlayProvider
from duguai.game.robot import Robot


class RandomAgent(Robot.Agent):
    """随机挑选一个动作的策略"""

    def exec(self, state_vector: Union[np.ndarray, List[int]], action_list: List[int]) -> int:
        """
        随机挑选一个动作的策略
        @param state_vector: 状态向量
        @param action_list: 动作
        """
        return sample(action_list, 1)[0]


class AbstractQLAgent(Robot.Agent, ABC):
    """
    抽象Q-Learning智能体
    """

    def __init__(self, play_q_table: np.ndarray, follow_q_table: np.ndarray):
        self._play_q_table = play_q_table
        self._follow_q_table = follow_q_table

    def _get_q_table1_state1(self, state_vector1: Union[np.ndarray, List[int]]) -> Tuple[np.ndarray, int]:
        if len(state_vector1) == PlayQLHelper.STATE_VECTOR_SIZE:
            return self._play_q_table, PlayQLHelper.state_to_int(state_vector1)
        return self._follow_q_table, FollowQLHelper.state_to_int(state_vector1)


class QLExecuteAgent(AbstractQLAgent):
    """
    不训练Q表，利用现有Q表执行行动的智能体
    """

    def __init__(self, play_q_table: np.ndarray, follow_q_table: np.ndarray):
        super().__init__(play_q_table, follow_q_table)

    def exec(self, state_vector1: Union[np.ndarray, List[int]], actions1: List[int]) -> int:
        """
        根据状态向量和动作直接查询Q表执行
        @param state_vector1: 状态向量
        @param actions1: 动作数组
        @return: Q值最大的动作
        """
        q_table1, state1 = self._get_q_table1_state1(state_vector1)
        if mode == 'debug':
            if len(state_vector1) == PlayQLHelper.STATE_VECTOR_SIZE:
                for a in actions1:
                    logging.info(PlayProvider.ActionProvider.ACTION_VIEW[a] + ' ' + str(q_table1[state1, a]))
            else:
                for a in actions1:
                    logging.info(FollowProvider.ACTION_VIEW[a] + ' ' + str(q_table1[state1, a]))
            logging.info('-------------------------------')

        q_list = q_table1[state1, actions1]
        max_q = np.max(q_list)
        approximate_max_actions = np.array(actions1)[q_list + 0.1 >= max_q]
        return np.random.choice(approximate_max_actions, 1)[0]


class QLTrainingAgent(AbstractQLAgent):
    """
    训练Q-learning算法的智能体
    Q(S, A) := Q(S, A) + alpha * [(R + gamma * max Q(S', a) - Q(S, A)]
    @note: 该类不负责持久化保存训练完的Q表
    """

    def __init__(self, play_q_table: np.ndarray, follow_q_table: np.ndarray, alpha: float, gamma: float,
                 epsilon: float = 0.1):
        super().__init__(play_q_table, follow_q_table)

        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon

        self.action0: int = -1
        self.state0: int = -1
        self.reward0: int = -1
        self.q_table0: Optional[np.ndarray] = None

    def _epsilon_greedy(self, q_table: np.ndarray, actions: List[int], state: int) -> int:
        """
        epsilon-贪心法。
        大多数时候(1-epsilon)的概率挑选最优动作A_t := argmax Q_t(a)
        有epsilon的概率随机挑选一个状态
        """
        if random() < self._epsilon:
            return sample(actions, 1)[0]
        q_list = q_table[state, actions]
        max_a = np.max(q_list)
        max_actions = np.array(actions)[q_list == max_a]
        return np.random.choice(max_actions, 1)[0]

    def _update_q_table0(self, state1: int, actions1: List[int], q_table1: np.ndarray):
        """使用Q-Learning算法更新Q表"""
        q_value0 = self.q_table0[self.state0, self.action0]
        self.q_table0[self.state0, self.action0] += self._alpha * (
                self.reward0 + self._gamma * (np.max(q_table1[state1, actions1]) - q_value0)
        )

    def update_game_over(self, reward: int) -> None:
        """游戏结束时，更新Q表"""
        q_value0 = self.q_table0[self.state0, self.action0]
        self.q_table0[self.state0, self.action0] += self._alpha * (reward - q_value0)

    def _update_reward0(self, actions1: List[int], state_vector1):
        if len(state_vector1) == PlayQLHelper.STATE_VECTOR_SIZE:
            if self.action0 in PlayProvider.ActionProvider.BAD_ACTION:
                self.reward0 = (-1 - len(actions1) * 0.1)
                return
        elif self.action0 in FollowProvider.BAD_ACTION:
            self.reward0 = (-1 - len(actions1) * 0.1)
            return

        self.reward0 = -1

    def exec(self, state_vector1: Union[np.ndarray, List[int]], actions1: List[int]) -> int:
        """
        执行Q-Learning算法
        @param state_vector1: 状态向量
        @param actions1: 动作
        @return: 通过Q-Learning选择出来的动作
        """
        q_table1, state1 = self._get_q_table1_state1(state_vector1)
        if self.q_table0 is not None:
            self._update_q_table0(state1, actions1, q_table1)

        self.q_table0 = q_table1
        self.state0 = state1
        self.action0 = self._epsilon_greedy(q_table1, actions1, state1)
        self._update_reward0(actions1, state_vector1)
        return self.action0


class AbstractQLHelper(metaclass=ABCMeta):
    """
    Q-Learning辅助类
    """

    def __init__(self, state_vector: Union[List[int], np.ndarray], actions: List[int]):
        self._actions = actions
        self._state = self.state_to_int(state_vector)

    @classmethod
    @abstractmethod
    def state_to_int(cls, vector: Union[List[int], np.ndarray]) -> int:
        """
        将状态向量转换为一个整型数，便于查询Q表
        """
        pass


class FollowQLHelper(AbstractQLHelper):
    """跟牌时Q-Learning的辅助类"""

    STATE_LEN = 20736
    ACTION_LEN = 9

    WEIGHT = (3456, 1152, 384, 48, 6, 1)

    @classmethod
    def state_to_int(cls, vector: Union[List[int], np.ndarray]) -> int:
        """
        状态向量的每一个取值范围分别为0到 5，2，2，7，7，5
        @return: [0, 20735]中的一个整数
        """
        return sum(vector[i] * cls.WEIGHT[i] for i in range(6))


class PlayQLHelper(AbstractQLHelper):
    """出牌时Q-Learning的辅助类"""

    __N_MAP = {0: 0, 1: 1, 2: 3, 3: 6}
    __V_WEIGHT_MAP = {11: 1,
                      10: 8,
                      9: 8 * 8,
                      8: 8 * 8 * 3,
                      7: 8 * 8 * 3 * 2,
                      6: 8 * 8 * 3 * 2 * 3,
                      5: 8 * 8 * 3 * 2 * 3 * 2,
                      4: 8 * 8 * 3 * 2 * 3 * 2 * 3,
                      1: 8 * 8 * 3 * 2 * 3 * 2 * 3 * 3,
                      0: 8 * 8 * 3 * 2 * 3 * 2 * 3 * 3 * 6}

    STATE_VECTOR_SIZE = 12

    STATE_LEN = 1244160
    ACTION_LEN = 17

    @classmethod
    def state_to_int(cls, vector: Union[List[int], np.ndarray]) -> int:
        """
        (4+3+2+1)*(3+2+1)*3^2*2*3*2*3*8*8 = 1244160
        @return: [0, 1244159]中的一个整数
        """

        n0 = vector[0] + cls.__N_MAP[vector[1]]
        n1 = vector[2] + cls.__N_MAP[vector[3]]

        return n0 * cls.__V_WEIGHT_MAP[0] + n1 * cls.__V_WEIGHT_MAP[1] + sum(
            vector[i] * cls.__V_WEIGHT_MAP[i] for i in range(4, 12))


def load_q_table(file_name: str, row: int, col: int) -> np.ndarray:
    """
    从.npy文件中加载Q表。若不存在，直接根据row和col返回一个初始化Q表。
    @param file_name: 文件名字符串，后缀为.npy
    @param row: Q表的行数
    @param col: Q表的列数
    @return: Q表
    """
    if os.path.exists(file_name):
        q_table: np.ndarray = np.load(file_name, allow_pickle=True)
    else:
        logging.info('找不到文件，初始化一个全为0的, shape为({}, {})的Q表'.format(row, col))
        q_table: np.ndarray = np.zeros((row, col))

    if q_table.shape != (row, col):
        raise ValueError('行数和列数错误')

    logging.info('加载成功')
    logging.info(q_table)
    return q_table


def save_q_table(file_name: str, q_table: np.ndarray):
    """
    保存Q表
    @param file_name: 文件名
    @param q_table: 待保存的Q表
    """
    np.save(file_name, q_table)
    logging.info(q_table)
    logging.info('保存成功')

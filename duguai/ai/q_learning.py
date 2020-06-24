# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from random import sample
from typing import List, Union

import numpy as np

from game.robot import Robot


class RandomActionPolicy(Robot.ActionPolicy):
    """随机挑选一个动作的策略"""

    def pick(self, state_vector: Union[np.ndarray, List[int]], action_list: List[int]) -> int:
        """
        随机挑选一个动作的策略
        @param state_vector: 状态向量
        @param action_list: 动作
        """
        return sample(action_list, 1)[0]


class AbstractQLTrainer(metaclass=ABCMeta):

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


class FollowQLTrainer(AbstractQLTrainer):
    STATE_LEN = 11664
    ACTION_LEN = 9

    @classmethod
    def state_to_int(cls, vector: Union[List[int], np.ndarray]) -> int:
        """
        状态向量的每一个取值范围分别为0到 5，2，2，5，5，5
        @return: [0, 11663]中的一个整数
        """
        return vector[-6] * 1944 + vector[-5] * 648 + vector[-4] * 216 + vector[-3] * 36 + vector[-2] * 6 + vector[-1]


class PlayQLTrainer(AbstractQLTrainer):
    n_map = {0: 0, 1: 1, 2: 3, 3: 6}

    v_weight_map = {11: 1, 10: 6, 9: 36, 8: 108, 7: 216, 6: 648, 5: 1296, 4: 3888}

    STATE_LEN = 699840
    ACTION_LEN = 16
    action_map = {1: 0, 2: 4, 3: 7, 4: 9, 5: 12, 6: 15, 7: 16}

    @classmethod
    def state_to_int(cls, vector: Union[List[int], np.ndarray]) -> int:
        """
        (4+3+2+1)*(3+2+1)*3^2*2*3*2*3*6*6 = 699840
        @return: [0, 699839]中的一个整数
        """

        n0 = vector[0] + cls.n_map[vector[1]]
        n1 = vector[2] + cls.n_map[vector[3]]

        return n0 * 69984 + n1 * 11664 + sum(vector[i] * cls.v_weight_map[i] for i in range(4, 12))

    def map_action(self, action: int) -> int:
        return self.action_map[action] + action // 10

# -*- coding: utf-8 -*-
"""
AI进行斗地主操作的类，多种算法在此集成
@author 江胤佐
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Iterator, Union, Tuple, List

import numpy as np

from ai import process
from ai.call_landlord import get_svc, z_score
from ai.executor import execute_play, execute_follow
from ai.provider import PlayProvider, FollowProvider
from game.game_env import GameEnv, _remove_last_combo


class Robot(GameEnv.AbstractPlayer):
    """
    AI，由机器学习算法决定操作
    @author 江胤佐
    """

    def __init__(self, game_env: GameEnv, order: int, action_policy: Robot.ActionPolicy):
        super().__init__(game_env, order)
        self.svc = get_svc()
        self.play_provider = PlayProvider(order)
        self.follow_provider = FollowProvider(order)
        self.__landlord_id: int = 0
        self._action_policy: Robot.ActionPolicy = action_policy

    def update_msg(self, msgs: Union[Iterator, str]) -> None:
        """Robot always ignores text message."""
        pass

    def update_last_combo(self, is_play: bool) -> None:
        """用于更新Q表"""
        # TODO
        pass

    def update_game_over(self, victor: Union[Tuple[int], int]) -> None:
        """用于更新Q表"""
        # TODO
        pass

    def call_landlord(self) -> bool:
        """
        AI叫地主
        @return: 叫: True; 不叫: False
        """
        return self.svc.predict(z_score([process(self.hand), ])) == 1

    def update_landlord(self, landlord_id: int) -> None:
        """
        AI保存地主玩家ID
        """
        self.__landlord_id = landlord_id
        self.play_provider.add_landlord_id(landlord_id)
        self.follow_provider.add_landlord_id(landlord_id)

    class ActionPolicy(metaclass=ABCMeta):
        """动作挑选的策略接口"""

        @abstractmethod
        def pick(self, state_vector: Union[np.ndarray, List[int]], action_list: List[int]) -> int:
            """
            根据状态向量挑选一个动作
            @param state_vector: 状态向量
            @param action_list: 动作
            @return: 从action_list中挑选出来的动作
            """
            pass

    @_remove_last_combo
    def follow(self) -> None:
        """
        AI跟牌
        """
        state, bombs, good_actions, max_actions, action_list = self.follow_provider.provide(
            last_combo_owner_id=self.game_env.last_combo_owner_id,
            hand_p=self.game_env.hand_p,
            hand_n=self.game_env.hand_n,
            cards=self.hand,
            last_combo=deepcopy(self.game_env.last_combo))

        action: int = self._action_policy.pick(state, action_list)
        self.last_combo.cards = execute_follow(action, bombs, good_actions, max_actions)
        if not self.valid_follow():
            raise ValueError('AI跟牌不合法, AI出的牌: {}, 上一次牌: {}'
                             .format(self.last_combo.cards_view, self.game_env.last_combo))

    @_remove_last_combo
    def play(self) -> None:
        """
        AI出牌
        """
        play_hand, state_vector, action_list = self.play_provider.provide(
            self.hand,
            hand_p=self.game_env.hand_p,
            hand_n=self.game_env.hand_n)
        action: int = self._action_policy.pick(state_vector, action_list)
        self.last_combo.cards = execute_play(play_hand, action)
        if not self.last_combo.is_valid():
            raise ValueError('AI出牌非法, AI出的牌: {}'.format(self.last_combo.cards_view))

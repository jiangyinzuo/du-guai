# -*- coding: utf-8 -*-
"""
AI进行斗地主操作的类，多种算法在此集成
@author 江胤佐
"""
from copy import deepcopy

from ai import process
from ai.call_landlord import get_svc, z_score
from ai.executor import execute_play, execute_follow
from ai.provider import PlayProvider, FollowProvider
from ai.q_learning import get_play_action, get_follow_action
from card.cards import cards_view
from duguai import mode
from game.game_env import GameEnv, _remove_last_combo


class Robot(GameEnv.AbstractPlayer):
    """
    AI，由机器学习算法决定操作
    @author 江胤佐
    """

    def __init__(self, game_env: GameEnv, order: int):
        super().__init__(game_env, order)
        self.svc = get_svc()
        self.play_provider = PlayProvider(order)
        self.follow_provider = FollowProvider(order)
        self.__landlord_id: int = 0

    def call_landlord(self) -> bool:
        """
        AI叫地主
        @return: 叫: True; 不叫: False
        """
        if mode == 'debug':
            print('AI{}的手牌:'.format(self.order), cards_view(self.hand))
        return self.svc.predict(z_score([process(self.hand), ])) == 1

    def notify_landlord(self, landlord_id: int) -> None:
        """
        AI保存地主玩家ID
        """
        self.__landlord_id = landlord_id
        self.play_provider.add_landlord_id(landlord_id)
        self.follow_provider.add_landlord_id(landlord_id)

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
            last_combo=deepcopy(self.game_env.former_combo))

        action: int = get_follow_action(state, action_list)
        self.last_combo.cards = execute_follow(action, bombs, good_actions, max_actions)
        if not self.valid_follow():
            raise ValueError('AI跟牌不合法, AI出的牌: {}, 上一次牌: {}'
                             .format(self.last_combo.cards_view, self.game_env.former_combo))

    @_remove_last_combo
    def play(self) -> None:
        """
        AI出牌
        """
        play_hand, state_vector, action_list = self.play_provider.provide(
            self.hand,
            hand_p=self.game_env.hand_p,
            hand_n=self.game_env.hand_n)
        action: int = get_play_action(state_vector, action_list)
        self.last_combo.cards = execute_play(play_hand, action)
        if not self.last_combo.is_valid():
            raise ValueError('AI出牌非法, AI出的牌: {}'.format(self.last_combo.cards_view))

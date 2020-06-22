# -*- coding: utf-8 -*-
from ai import process
from ai.call_landlord import get_svc, z_score
from ai.executor import Executor
from ai.provider import AbstractProvider
from ai.q_learning import get_action
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
        self.provider = AbstractProvider(order)
        self.__landlord_id: int = 0
        self.__executor = Executor()

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
        self.provider.add_landlord_id(landlord_id)

    @_remove_last_combo
    def follow(self) -> None:
        """
        AI跟牌
        """
        state, actions, good_hand, bad_hand = self.provider.provide(
            self.hand,
            hand_p=self.game_env.hand_p,
            hand_n=self.game_env.hand_n,
            last_combo_owner=self.game_env.last_combo_owner,
            last_combo=self.game_env.former_combo)
        action: int = get_action(state, actions)
        self.last_combo.cards = self.__executor.execute(action, good_hand, bad_hand, self.game_env.former_combo)
        if not self.valid_follow():
            raise ValueError('AI跟牌不合法')

    @_remove_last_combo
    def play(self) -> None:
        """
        AI出牌
        """
        state, actions, good_hand, _ = self.provider.provide(
            self.hand,
            hand_p=self.game_env.hand_p,
            hand_n=self.game_env.hand_n,
            last_combo_owner=self.game_env.last_combo_owner)
        action: int = get_action(state, actions)
        self.last_combo.cards = self.__executor.execute(action, good_hand)
        if not self.last_combo.is_valid():
            raise ValueError('AI跟牌非法')

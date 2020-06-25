# -*- coding: utf-8 -*-
"""
人类玩家模块
@author: 江胤佐
"""
from typing import Iterator, Union, Set

from card.cards import cards_view
from duguai.game.game_env import GameEnv, _remove_last_combo, SPLIT_LINE
from utils import is_in


class Human(GameEnv.AbstractPlayer, GameEnv.MessageObserver):
    """
    人类玩家，由控制台输入输出进行操作
    @author 江胤佐
    """

    def __init__(self, game_env: GameEnv, order: int):
        super().__init__(game_env, order)

    def update_msg(self, msgs: Union[Iterator, str]) -> None:
        """
        人类玩家收到GameEnv对象发来的消息
        @param msgs: 消息
        """
        if isinstance(msgs, Iterator):
            for msg in msgs:
                print(msg)
        else:
            print(msgs)

    def update_last_combo(self) -> None:
        """
        GameEnv更新了上一次出牌操作
        """
        if self.game_env.last_combo_owner_id == self.game_env.turn:
            print(self.game_env.user_info(0) +
                  '打出了' +
                  self.game_env.last_combo.cards_view)
        else:
            print(self.game_env.user_info(0) + '空过')
        print(SPLIT_LINE)

    def update_game_over(self, victors: Set[int]) -> None:
        """
        GameEnv通知玩家游戏结束
        @param victors: 胜利者
        """
        print('玩家', victors, '获胜')

    def call_landlord(self) -> bool:
        """
        玩家叫地主
        @return: 叫: True; 不叫: False
        """
        print('玩家{}的手牌:'.format(self.order), cards_view(self.hand))
        return input('>>> (输入1叫地主, 输入其它键不叫地主)') == '1'

    def update_landlord(self, landlord_id: int) -> None:
        """
        通知人类玩家，谁成为了地主
        """
        print(SPLIT_LINE)
        print('玩家{}叫了地主'.format(landlord_id))
        print('地主获得了3张牌: {}'.format(cards_view(self.game_env.cards[3])))
        print(SPLIT_LINE)

    @_remove_last_combo
    def follow(self) -> None:
        """
        玩家跟牌
        """
        while True:
            cards_v = input('你的手牌: {}\n上家 {} 手牌数量: {}\n下家 {} 手牌数量: {}\n>>> (输入要出的牌，以空格分隔。直接回车代表空过。)'
                            .format(cards_view(self.hand),
                                    self.game_env.user_info(-1),
                                    self.game_env.hand_p,
                                    self.game_env.user_info(1),
                                    self.game_env.hand_n)).upper()
            self.last_combo.cards_view = cards_v
            if self.valid_follow():
                break
            else:
                print('输入非法!')

    @_remove_last_combo
    def play(self) -> None:
        """
        玩家出牌
        """
        while True:
            self.last_combo.cards_view = input('你的手牌:{}\n>>> (输入要出的牌，以空格分隔。)'.format(cards_view(self.hand)))
            if is_in(self.last_combo.cards, self.hand) and self.last_combo.is_not_empty():
                break
            else:
                print('输入非法!')

# -*- coding: utf-8 -*-
from card.cards import cards_view
from game.game_env import GameEnv, _remove_last_combo
from utils import is_in


class Human(GameEnv.AbstractPlayer):
    """
    人类玩家，由控制台输入输出进行操作
    @author 江胤佐
    """

    def __init__(self, game_env: GameEnv, order: int):
        super().__init__(game_env, order)

    def call_landlord(self) -> bool:
        """
        玩家叫地主
        @return: 叫: True; 不叫: False
        """
        print('玩家{}的手牌:'.format(self.order), cards_view(self.hand))
        return input('>>> (输入1叫地主, 输入其它键不叫地主)') == '1'

    def notify_landlord(self, landlord_id: int) -> None:
        """
        通知人类玩家，谁成为了地主
        """
        print('玩家{}叫了地主'.format(landlord_id))

    @_remove_last_combo
    def follow(self) -> None:
        """
        玩家跟牌
        """
        while True:
            cards_v = input('你的手牌:{}\n>>> (输入要出的牌，以空格分隔。直接回车代表空过。)'.format(cards_view(self.hand)))
            self.last_combo.cards_view = cards_v
            if not cards_v or is_in(self.last_combo.cards, self.hand) and \
                    self.last_combo.is_valid() and \
                    self.last_combo > self.game_env.former_combo:
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

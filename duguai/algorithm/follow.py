# -*- coding: utf-8 -*-
"""
跟牌算法
"""
import numpy as np

from card_def import CardType


def _single_type_eval(last_cards: np.ndarray, cur_cards: np.ndarray):
    pass


_type_eval_funcs: dict = {
    CardType.SINGLE: _single_type_eval
}


class FollowEvaluator:
    """
    跟牌策略评估器，用于跟牌
    """

    def get_valid_follows(self,
                          last_cards: np.ndarray,
                          cur_cards: np.ndarray):
        """
        获取当前出牌玩家所有合法的跟牌组合
        :param last_cards: 上一次出的牌
        :param cur_cards: 本回合玩家的牌
        :return:
        """
        pass

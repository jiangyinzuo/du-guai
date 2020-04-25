# -*- coding: utf-8 -*-
"""
叫地主算法
"""
from copy import deepcopy

from card_def import *


class LandlordClassifier:
    """
    叫地主分类器，根据得到的牌堆判断是否应该叫地主
    """

    def __init__(self, cards: np.ndarray):
        """
        :param cards: AI获得的牌，共17张
        """
        if cards.shape[0] != 17:
            raise ValueError('初始牌堆应该是17张')
        self.__cards = cards
        self.__cards.sort()

        self.__ghost = 0
        self.__fit_ghost()

        self.__card_2_count = int(np.sum(self.__cards == CARD_2, dtype=int))

        # ------- 求组合相关变量 --------
        self.__combo_count = 0
        self.__cards_count = [0, ] * 15
        for i in self.__cards:
            self.__cards_count[i] += 1

    def __fit_ghost(self):
        if CARD_G0 in self.__cards[-2:]:
            self.__ghost += 1
        if self.__cards[-1] == CARD_G1:
            self.__ghost += 2

    def __get_min_combo(self):
        # TODO 测试该方法
        temp_cards_count: list = deepcopy(self.__cards_count)
        continue_count: int = 0
        min_card_count: int = 0
        # 遍历3 到 A
        for i in range(CARD_3, CARD_2):
            min_card_count = min(min_card_count, temp_cards_count[i])
            if min_card_count != 0:
                continue_count += 1
                if continue_count >= 5:
                    min_card_count = 0
                    for j in range(0, continue_count):
                        temp_cards_count[i - j] -= min_card_count
                        min_card_count = min(min_card_count, temp_cards_count[i - j])
                    self.__combo_count += 1
            elif continue_count < 5:
                continue_count = 0
            elif self.__cards_count[i-1] >= 3:
                temp_cards_count[i-1] = self.__cards_count[i-1]
        for i in range(CARD_3, CARD_2):
            if 1 <= temp_cards_count[i] <= 2:
                self.__combo_count += 1

    def call_landlord(self) -> bool:
        """
        AI叫地主
        :return: 叫: True; 不叫: False
        """
        if self.__ghost + self.__card_2_count >= 5:
            return True
        # TODO 叫地主
        return False

    @property
    def cards(self) -> np.ndarray:
        """
        牌堆
        :return: 排好序的牌
        """
        return self.__cards

    @property
    def ghost(self) -> int:
        """
        大王小王属性
        :return: 0: 没有王; 1: 有小王; 2: 有大王; 3: 有大王小王
        """
        return self.__ghost

    @property
    def card_2_count(self) -> int:
        """
        牌组中2的个数
        :return: 2的个数，在[0, 4]中
        """
        return self.__card_2_count

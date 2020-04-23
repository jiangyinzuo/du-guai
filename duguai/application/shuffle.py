# -*- coding: utf-8 -*-
import numpy as np


def shuffle() -> np.ndarray:
    """
    洗牌
    """
    cards: np.ndarray = np.asarray([card for card in range(1, 14)] * 4 + [15, 16], dtype=int)
    np.random.shuffle(cards)
    return cards

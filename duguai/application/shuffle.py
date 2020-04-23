# -*- coding: utf-8 -*-
import numpy as np


def shuffle(count: int = 54) -> np.ndarray:
    """
    洗牌
    """
    if 1 <= count <= 54:
        cards: np.ndarray = np.asarray([card for card in range(1, 14)] * 4 + [15, 16], dtype=int)
        np.random.shuffle(cards)
        return cards[:count]
    raise ValueError('count of cards must in interval [1, 54]')

import numpy as np
def shuffle(count: int = 54):
    if 1 <= count <= 54:
        cards = np.asarray([card for card in range(1, 14)] * 4 + [14, 15], dtype=int)
        np.random.shuffle(cards)
        allo=(np.sort(cards[0:17]), np.sort(cards[17:34]), np.sort(cards[34:51]), np.sort(cards[51:54]))
        return allo
    raise ValueError('count of cards must in interval [1, 54]')

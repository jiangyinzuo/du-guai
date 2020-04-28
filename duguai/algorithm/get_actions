import numpy as np
def get_actions(next_moves, actions_lookuptable, game):
    """
    0-14: 单出， 1-13，小王，大王
    15-27: 对，1-13
    28-40: 三，1-13
    41-196: 三带1，先遍历111.2，111.3，一直到131313.12
    197-352: 三带2，先遍历111.22,111.33,一直到131313.1212
    353-366: 炸弹，1111-13131313，加上王炸
    367-402: 先考虑5个的顺子，按照顺子开头从小到达进行编码，共计8+7+..+1=36
    430: yaobuqi
    429: buyao
    """
    actions = np.array([])
    for cards in next_moves:
        key = []
        for card in cards:
            key.append(int(card))
        key.sort()
        actions.append(actions_lookuptable[str(key)])
    
    #yaobuqi
    if len(actions) == 0:
        actions.append(430)
    #buyao
    elif game.last_move != "start":
        actions.append(429)
        
    return actions

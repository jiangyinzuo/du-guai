import copy
import time
from ..mcts.mcts import MCTS
from ..mcts.tree_policies import UCB1
from ..mcts.default_policies import random_terminal_roll_out
from ..mcts.backups import monte_carlo
import numpy as np
from shuffle import shuffle
from call_landlord import LandlordClassifier
from player import Player

# 发牌
def game_init(players, playrecords, cards, train):
        #if train:
        # 洗牌
        cards=shuffle()
        c1=cards[0:17]
        c2=cards[17:34]
        c3=cards[34:51]
        if LandlordClassifier(c1).call_landlord():
            c1+=cards[51:54]
        elif LandlordClassifier(c2).call_landlord():
            c2+=cards[51:54]
        elif LandlordClassifier(c3).call_landlord():
            c3+=cards[51:54]
        else:
            c1+=cards[51:54]
        p1=Player(c1.sort(), 'p1')
        p2=Player(c2.sort(), 'p2')
        p3=Player(c3.sort(), 'p3')
        
        #players[0].cards_left = playrecords.cards_left1 = p1_cards
        #players[1].cards_left = playrecords.cards_left2 = p2_cards
        #players[2].cards_left = playrecords.cards_left3 = p3_cards
    '''
    else:
        # 洗牌
        np.random.shuffle(cards.cards)
        # 排序
        p1_cards = cards.cards[:20]
        p1_cards.sort(key=lambda x: x.rank)
        p2_cards = cards.cards[20:37]
        p2_cards.sort(key=lambda x: x.rank)
        p3_cards = cards.cards[37:]
        p3_cards.sort(key=lambda x: x.rank)
        players[0].cards_left = playrecords.cards_left1 = p1_cards
        players[1].cards_left = playrecords.cards_left2 = p2_cards
        players[2].cards_left = playrecords.cards_left3 = p3_cards
    '''


# 展示扑克函数
def card_show(cards, info, n):
    # 扑克牌记录类展示
    if n == 1:
        print(info)
        names = []
        for i in cards:
            names.append(i.name + i.color)
        print(names)
    # Moves展示
    elif n == 2:
        if len(cards) == 0:
            return 0
        print(info)
        moves = []
        for i in cards:
            names = []
            for j in i:
                names.append(j.name + j.color)
            moves.append(names)
        print(moves)
    # record展示
    elif n == 3:
        print(info)
        names = []
        for i in cards:
            tmp = [i[0]]
            tmp_name = []
            # 处理要不起
            try:
                for j in i[1]:
                    tmp_name.append(j.name + j.color)
                tmp.append(tmp_name)
            except:
                tmp.append(i[1])
            names.append(tmp)
        print(names)


# 在Player的next_moves中选择出牌方法
def choose(next_move_types, next_moves, last_move_type, last_move, cards_left, model, RL, my_config, game, player_id,
           action):
    if model == "mcts":
        # init mcts
        if action == "mcts":
            # 要不起不需要mcst
            if len(next_moves) == 0:
                print("actions", [430])
                return "yaobuqi", []

            game_copy = copy.deepcopy(game)

            game_copy.players[0].model = "mcts"
            game_copy.players[1].model = "random"
            game_copy.players[2].model = "random"

            mcts = MCTS(tree_policy=UCB1(c=1.41),
                        default_policy=random_terminal_roll_out,
                        backup=monte_carlo,
                        game=game_copy)

            # state
            # TODO s = get_state()
            # action
            # TODO actions = get_actions()
            # new state
            # TODO s = combine(s, actions)

            begin = time.time()
            best_action, win_pob = mcts(s, n=2000)
            duration = time.time() - begin
            print("actions", actions, "best_action", best_action,
                  "win_pob", win_pob, "time", duration)

            if best_action == 429:
                return "buyao", []
            elif best_action == 430:
                return "yaobuqi", []
            else:
                # TODO best_action_id = actions.index(best_action)
                return next_move_types[best_action_id], next_moves[best_action_id]
        # mcts simulation
        else:
            if action == 429:
                return "buyao", []
            elif action == 430:
                return "yaobuqi", []
            else:
                return next_move_types[action], next_moves[action]

    # 训练model
    elif model == "rl":
        if action[3][action[2]] == 429:
            return "buyao", []
        elif action[3][action[2]] == 430:
            return "yaobuqi", []
        else:
            return action[0][action[2]], action[1][action[2]]

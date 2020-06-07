import copy
import time

from ..mcts.backups import monte_carlo
from ..mcts.default_policies import random_terminal_roll_out
from ..mcts.mcts import MCTS
from ..mcts.tree_policies import UCB1


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
            except Exception:
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

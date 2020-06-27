# -*- coding: utf-8 -*-
"""
斗地主程序的入口文件。
"""
import logging
import os
import sys

sys.path.append('..')

if __name__ == '__main__':
    from duguai.ai.q_learning import QLExecuteAgent, load_q_table, PlayQLHelper, FollowQLHelper
    from duguai.game.human import Human
    from duguai.game.robot import Robot
    from duguai.logger import log_locals
    from duguai import mode, test, play_dataset, follow_dataset
    from duguai.game.game_env import GameEnv

    if not os.path.isfile(play_dataset) or not os.path.isfile(follow_dataset):
        print('找不到数据集', play_dataset, follow_dataset)
        exit(0)

    if mode == 'debug':
        logging.basicConfig(level=logging.DEBUG)

    play_q_table = load_q_table(play_dataset, PlayQLHelper.STATE_LEN, PlayQLHelper.ACTION_LEN)
    follow_q_table = load_q_table(follow_dataset, FollowQLHelper.STATE_LEN,
                                  FollowQLHelper.ACTION_LEN)

    game_env = GameEnv()
    try:
        if test == 'on':
            robot0 = Robot(game_env, QLExecuteAgent(play_q_table, follow_q_table), 'ql0')
            robot1 = Robot(game_env, QLExecuteAgent(play_q_table, follow_q_table), 'ql1')
            robot2 = Robot(game_env, QLExecuteAgent(play_q_table, follow_q_table), 'ql2')
            game_env.add_players(robot0, robot1, robot2)
            for i in range(100):
                game_env.start()
            print('success!')
        else:
            human = Human(game_env, 'human')
            robot1 = Robot(game_env, QLExecuteAgent(play_q_table, follow_q_table), 'ql1')
            robot2 = Robot(game_env, QLExecuteAgent(play_q_table, follow_q_table), 'ql2')
            game_env.add_players(human, robot1, robot2)

            game_env.start()
            for p in (human, robot1, robot2):
                print(p.victory_count)
    except EOFError as e:
        game_env.notify(GameEnv.U_MSG, msgs='Bye~')
        exit(0)
    except KeyboardInterrupt as e:
        game_env.notify(GameEnv.U_MSG, msgs='Bye~')
        exit(0)
    except Exception as e:
        logging.exception(e)
        if mode == 'debug':
            log_locals(e)
        exit(0)

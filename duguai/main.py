# -*- coding: utf-8 -*-
"""
斗地主程序的入口文件。
"""
import logging
import sys

sys.path.append('..')

if __name__ == '__main__':
    from duguai.ai.q_learning import RandomAgent, QLExecuteAgent, load_q_table, PlayQLHelper, FollowQLHelper
    from duguai.game.human import Human
    from duguai.game.robot import Robot
    from duguai.logger import log_locals
    from duguai import mode, test
    from duguai.game.game_env import GameEnv

    game_env = GameEnv()
    try:
        play_q_table = load_q_table('../script/play_q_table.npy', PlayQLHelper.STATE_LEN, PlayQLHelper.ACTION_LEN)
        follow_q_table = load_q_table('../script/follow_q_table.npy', FollowQLHelper.STATE_LEN,
                                      FollowQLHelper.ACTION_LEN)
        if test == 'on':
            robot0 = Robot(game_env, 0, RandomAgent())
            robot1 = Robot(game_env, 1, QLExecuteAgent(play_q_table, follow_q_table))
            robot2 = Robot(game_env, 2, QLExecuteAgent(play_q_table, follow_q_table))
            game_env.add_players(robot0, robot1, robot2)
            for i in range(100):
                game_env.start()
            print('success!')
        else:
            human = Human(game_env, 0)
            robot1 = Robot(game_env, 1, QLExecuteAgent(play_q_table, follow_q_table))
            robot2 = Robot(game_env, 2, QLExecuteAgent(play_q_table, follow_q_table))
            game_env.add_players(human, robot1, robot2)

            game_env.start()
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

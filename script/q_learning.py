# -*- coding: utf-8 -*-
import logging
import sys

sys.path.append('..')

if __name__ == '__main__':
    from getopt import getopt, GetoptError
    from time import time

    from duguai import mode
    from duguai.ai.q_learning import load_q_table, save_q_table, PlayQLHelper, FollowQLHelper, QLTrainingAgent
    from duguai.game.game_env import GameEnv
    from duguai.game.robot import Robot
    from duguai.logger import log_locals

    train_times: int = 10
    try:
        opts, args = getopt(sys.argv[1:], 't:')
        for opt, arg in opts:
            if opt == '-t':
                train_times = int(arg)
                if train_times < 0 or train_times >= 100000000:
                    raise ValueError('train_times must be an integer between 1 and 99999999')
    except GetoptError as e:
        print('python q_learning.py -t <train_times>')
        sys.exit(2)
    except ValueError as e:
        print(e)
        sys.exit(2)
    except Exception as e:
        logging.exception(e)
        sys.exit(2)

    play_q_table = load_q_table('play_q_table.npy', PlayQLHelper.STATE_LEN, PlayQLHelper.ACTION_LEN)
    follow_q_table = load_q_table('follow_q_table.npy', FollowQLHelper.STATE_LEN, FollowQLHelper.ACTION_LEN)

    game_env = GameEnv()

    agent0 = QLTrainingAgent(play_q_table, follow_q_table, 0.5, 0.4)
    agent1 = QLTrainingAgent(play_q_table, follow_q_table, 0.5, 0.4)
    agent2 = QLTrainingAgent(play_q_table, follow_q_table, 0.5, 0.4)

    robot0 = Robot(game_env, 0, agent0)
    robot1 = Robot(game_env, 1, agent1)
    robot2 = Robot(game_env, 2, agent2)
    game_env.add_players(robot0, robot1, robot2)

    start_time = time()
    try:
        for i in range(train_times):
            game_env.start()
    except Exception as e:
        logging.exception(e)
        if mode == 'debug':
            log_locals(e)
    finally:
        logging.info('训练时间: %f 秒; 训练次数: %d' % ((time() - start_time), train_times))
        save_q_table('play_q_table.npy', play_q_table)
        save_q_table('follow_q_table.npy', follow_q_table)

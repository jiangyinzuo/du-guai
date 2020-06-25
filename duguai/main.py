# -*- coding: utf-8 -*-
"""
斗地主程序的入口文件。
"""
import logging

from ai.q_learning import RandomAgent
from duguai import mode, test
from duguai.game.game_env import GameEnv
from game.human import Human
from game.robot import Robot
from logger import log_locals

if __name__ == '__main__':
    game_env = GameEnv()
    try:
        if test == 'on':
            robot0 = Robot(game_env, 0, RandomAgent())
            robot1 = Robot(game_env, 1, RandomAgent())
            robot2 = Robot(game_env, 2, RandomAgent())
            game_env.add_players(robot0, robot1, robot2)
            for i in range(100):
                game_env.start()
            print('success!')
        else:
            human = Human(game_env, 0)
            robot1 = Robot(game_env, 1, RandomAgent())
            robot2 = Robot(game_env, 2, RandomAgent())
            game_env.add_players(human, robot1, robot2)
            for i in range(5):
                game_env.start()
    except Exception as e:
        logging.exception(e)
        if mode == 'debug':
            log_locals(e)
        exit(0)

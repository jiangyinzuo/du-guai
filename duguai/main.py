# -*- coding: utf-8 -*-
"""
斗地主程序的入口文件。
"""

from duguai.game.game_env import GameEnv, Human, Robot


if __name__ == '__main__':
    game_env = GameEnv()
    robot0 = Robot(game_env, 0)
    robot1 = Robot(game_env, 1)
    human2 = Human(game_env, 2)
    game_env.add_players(robot0, robot1, human2)
    game_env.start()

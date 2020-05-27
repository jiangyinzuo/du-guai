# -*- coding: utf-8 -*-
"""
斗地主程序的入口文件。
"""

from duguai.game.game_env import GameEnv, Human

if __name__ == '__main__':
    game_env = GameEnv()
    human0 = Human(game_env, 0)
    human1 = Human(game_env, 1)
    human2 = Human(game_env, 2)
    game_env.add_players(human0, human1, human2)
    game_env.start()

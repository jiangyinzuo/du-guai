# -*- coding: utf-8 -*-
"""
斗地主程序的入口文件。
"""

from duguai.game.game_env import GameEnv, Human

if __name__ == '__main__':
    game_env = GameEnv()
    game_env.add_players(Human(game_env, 0),
                         Human(game_env, 1),
                         Human(game_env, 2))
    game_env.start()

# -*- coding: utf-8 -*-

from duguai.game.game_env import GameEnv

game_env = GameEnv()


def test_game_env():
    result = game_env._GameEnv__shuffle()
    assert len(result) == 4
    assert len(result[0]) == len(result[1]) == len(result[2]) == 17
    assert len(result[3]) == 3

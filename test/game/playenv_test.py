# -*- coding: utf-8 -*-

from duguai.game.game_env import GameEnv

game_env = GameEnv()


def test_game_env():
    game_env.shuffle()
    assert len(game_env.cards) == 4
    assert len(game_env.cards[0]) == len(game_env.cards[1]) == len(game_env.cards[2]) == 17
    assert len(game_env.cards[3]) == 3

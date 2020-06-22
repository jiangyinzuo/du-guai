# -*- coding: utf-8 -*-
"""
斗地主程序的入口文件。
"""
import logging
import traceback

from duguai import mode
from duguai.game.game_env import GameEnv
from game.human import Human
from game.robot import Robot


def log_locals(err: Exception):
    """
    调试时输出本地变量的值
    @param err: 异常
    """
    flag = True
    for i in traceback.walk_tb(err.__traceback__):
        tb_frame = i[0]
        logging.error('co_name: {}; f_lineno: {}'.format(tb_frame.f_code.co_name, tb_frame.f_lineno))
        f_locals: dict = tb_frame.f_locals

        # 忽略第一个栈
        if flag:
            flag = False
            continue
        for k, v in f_locals.items():
            if hasattr(v, '__dict__'):
                logging.error('{}: {}'.format(k, v.__dict__))
            else:
                logging.error('{}: {}'.format(k, v))


if __name__ == '__main__':
    game_env = GameEnv()
    game_env.add_players(Human(game_env, 0),
                         Robot(game_env, 1),
                         Robot(game_env, 2))

    try:
        game_env.start()
    except Exception as e:
        logging.exception(e)
        if mode == 'debug':
            log_locals(e)

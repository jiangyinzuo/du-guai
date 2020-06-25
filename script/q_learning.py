# -*- coding: utf-8 -*-

import logging
import os

import numpy as np

from duguai import mode
from duguai.ai.q_learning import PlayQLHelper, FollowQLHelper, QLTrainingAgent
from game.game_env import GameEnv
from game.robot import Robot
from logger import log_locals


def load_q_table(file_name: str, row: int, col: int) -> np.ndarray:
    """
    从.npy文件中加载Q表。若不存在，直接根据row和col返回一个初始化Q表。
    @param file_name: 文件名字符串，后缀为.npy
    @param row: Q表的行数
    @param col: Q表的列数
    @return: Q表
    """
    if os.path.exists(file_name):
        q_table: np.ndarray = np.load(file_name, allow_pickle=True)
    else:
        logging.info('找不到文件，初始化一个全为0的, shape为({}, {})的Q表'.format(row, col))
        q_table: np.ndarray = np.zeros((row, col))

    if q_table.shape != (row, col):
        raise ValueError('行数和列数错误')

    logging.info('加载成功')
    print(q_table)
    return q_table


def save_q_table(file_name: str, q_table: np.ndarray):
    """
    保存Q表
    @param file_name: 文件名
    @param q_table: 待保存的Q表
    """
    np.save(file_name, q_table)
    print(q_table)
    logging.info('保存成功')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

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

    try:
        for i in range(3000):
            game_env.start()
            if i % 100 == 0:
                print('训练了%d次' % (i + 1))
    except Exception as e:
        logging.exception(e)
        if mode == 'debug':
            log_locals(e)
    finally:
        save_q_table('play_q_table.npy', play_q_table)
        save_q_table('follow_q_table.npy', follow_q_table)

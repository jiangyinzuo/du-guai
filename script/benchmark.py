# -*- coding: utf-8 -*-
"""
基准测试脚本
@author: 江胤佐
"""
import os
import sys
from getopt import getopt, GetoptError

sys.path.append('..')


def benchmark(play_q_table_path, follow_q_table_path):
    """
    基准测试
    """
    play_q_table = load_q_table(play_q_table_path, PlayQLHelper.STATE_LEN, PlayQLHelper.ACTION_LEN)
    follow_q_table = load_q_table(follow_q_table_path, FollowQLHelper.STATE_LEN,
                                  FollowQLHelper.ACTION_LEN)

    ql_agent0 = QLExecuteAgent(play_q_table, follow_q_table)
    random_agent1 = RandomAgent()

    game_env = GameEnv()

    robot0 = Robot(game_env, ql_agent0, 'ql')
    robot1 = Robot(game_env, random_agent1, 'rand1')
    robot2 = Robot(game_env, random_agent1, 'rand2')
    game_env.add_players(robot0, robot1, robot2)

    print('对战1000局')
    for i in range(1000):
        game_env.start()

    for r in (robot0, robot1, robot2):
        v1, v2 = r.victory_count
        print('{} 地主获胜场次: {}; 农民获胜场次: {}; 总计: {}'.format(r.name, v1, v2, v1 + v2))


if __name__ == '__main__':
    from duguai.game.robot import Robot
    from duguai.game.game_env import GameEnv
    from duguai.ai.q_learning import RandomAgent, load_q_table, PlayQLHelper, FollowQLHelper, QLExecuteAgent

    t = ''
    try:
        opts, args = getopt(sys.argv[1:], 't:')
        for opt, arg in opts:
            if opt == '-t':
                t = arg
    except GetoptError as e:
        print('python benchmark.py -t <train_times>')
        sys.exit(2)

    _play_q_table_path = '../dataset/play_q_table' + t + '.npy'
    _follow_q_table_path = '../dataset/follow_q_table' + t + '.npy'
    if os.path.isfile(_play_q_table_path) and os.path.isfile(_follow_q_table_path):
        print('训练了' + (t if t else '0') + '次的强化学习AI vs 随机决策AI')
        benchmark(_play_q_table_path, _follow_q_table_path)
    else:
        print('数据文件不存在')

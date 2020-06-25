# -*- coding: utf-8 -*-
"""
人工标记数据集的脚本
数据集格式：
一行代表一次叫地主结果。一行共18个数字，其中前17个数代表牌的大小，
最后一个数1代表叫地主，0代表不叫
"""
import sys

sys.path.append('..')

if __name__ == '__main__':
    from duguai.card.cards import cards_view
    from duguai.game.game_env import GameEnv

    g = GameEnv()

    with open('../notebook/call.csv', 'a') as f:
        while True:
            g.shuffle()
            print(cards_view(g.cards[0]))
            for i in g.cards[0]:
                f.write(str(i) + ',')
            t = input()
            if t == '-1':
                break
            f.write(t + '\n')

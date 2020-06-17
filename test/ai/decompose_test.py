# -*- coding: utf-8 -*-
from duguai.ai.decompose import *


def _actions_reward(state, actions):
    for action in actions:
        yield get_reward(state, action)


def test_move():
    test_list = [
        (
            [1, 2, 3, 4, 5, 6],
            [[2], [1, 2, 3, 4, 5]],
            1
        ),
        (
            [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
            [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6]],
            0
        ),
        (
            [1, 3, 3, 3, 4, 4, 4],
            [[3, 3, 4, 4, 4], [3, 3, 3, 4, 4, 4], [1, 3, 3, 3]],
            1
        )
    ]
    for i in test_list:
        state, actions, excepted_arg = i
        rewards = [r for r in _actions_reward(state, actions)]
        assert np.argmax(rewards) == excepted_arg


def test_seq_single_actions():
    assert len(get_seq_actions([4, 5, 6, 7], 2, 4)) == 1
    assert len(get_seq_actions([3, 4, 5, 6, 7, 8], 1, 5)) == 2


def test_single_actions():
    def _test(test_li, length):
        for i in test_li:
            state, excepted_arg = i

            rewards = [get_reward(state, a) for a in get_single_actions(state, length)]
            print(rewards)
            assert np.argmax(rewards) == excepted_arg

    test_list = [
        ([2, 3, 4, 4, 5, 6], 2),
        ([2, 3, 4, 4, 5, 5, 6, 6], 0),
        ([5, 5, 6, 6, 6, 7, 7, 8, 8], 1)
    ]

    _test(test_list, 1)

    test_list = [
        ([4, 5, 5, 6, 7, 8, 8, 8, 9, 10, 11], 1)
    ]

    _test(test_list, 2)


def test_combined_single_actions():
    test_list = [
        ([4, 5, 5, 6, 7, 8, 8, 8, 9, 10, 11], 1, 1, 1),
        ([3, 4, 5, 5, 6, 6, 7, 7, 7, 8, 9, 10, 11], 2, 2, 1)
    ]

    for i in test_list:
        state, ex_solo_args, ex_pair_args, pair_bt_solo = i
        solo_rewards = [get_reward(state, a) for a in get_single_actions(state, 1)]
        pair_rewards = [get_reward(state, a) for a in get_single_actions(state, 2)]
        print(solo_rewards, pair_rewards)
        assert np.argmax(solo_rewards) == ex_solo_args
        assert np.argmax(pair_rewards) == ex_pair_args
        assert (np.max(pair_rewards) - np.max(solo_rewards)) == pair_bt_solo


def test_get_good_actions():
    decomposer = Decomposer()
    print(decomposer.get_good_actions([9, 9]))
    print(decomposer.get_good_actions([5, 5, 6, 6, 7, 7, 7, 9, 9]))
    print(decomposer.get_good_actions([3, 4, 5, 5, 6, 6, 7, 7, 7, 8, 9, 10, 11]))

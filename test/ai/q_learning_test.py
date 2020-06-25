# -*- coding: utf-8 -*-
from duguai.ai.q_learning import PlayQLHelper, load_q_table, FollowQLHelper


def test_state_vector_to_int():
    assert PlayQLHelper.state_to_int([3, 3, 2, 2, 2, 2, 1, 2, 1, 2, 5, 5]) == 699839


def test_load_dataset():
    q_table = load_q_table('../../src/script/follow_q_table.npy', FollowQLHelper.STATE_LEN, FollowQLHelper.ACTION_LEN)
    assert q_table.shape == (FollowQLHelper.STATE_LEN, FollowQLHelper.ACTION_LEN)

# -*- coding: utf-8 -*-
from ai.q_learning import PlayQLHelper


def test_state_vector_to_int():
    assert PlayQLHelper.state_to_int([3, 3, 2, 2, 2, 2, 1, 2, 1, 2, 5, 5]) == 699839

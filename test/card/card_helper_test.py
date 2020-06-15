# -*- coding: utf-8 -*-
from ai.decompose import decompose_value


def test_decompose_value():
    test_list = [
        (6, [2, 3, 3, 3, 4, 4, 4, 5, 6]),
        (1, [2, 3, 4, 5, 7])
    ]
    for i in test_list:
        value = decompose_value(i[1])
        assert value == i[0]

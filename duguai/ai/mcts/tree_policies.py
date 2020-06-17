from __future__ import division

import warnings

import numpy as np

warnings.warn('暂时没用到，需要人维护', DeprecationWarning)


class UCB1(object):
    """
    The typical bandit upper confidence bounds ai.
    """

    def __init__(self, c):
        self.c = c

    def __call__(self, action_node):
        if self.c == 0:  # assert that no nan values are returned
            # for action_node.n = 0
            return action_node.q

        return (action_node.q +
                self.c * np.sqrt(2 * np.log(action_node.parent.n) /
                                 action_node.n))

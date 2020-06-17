from __future__ import division

import warnings

warnings.warn('暂时没用到，需要人维护', DeprecationWarning)


def monte_carlo(node):
    """
    A monte carlo update as in classical UCT.

    See feldman amd Domshlak (2014) for reference.
    :param node: The node to start the backup from
    """
    r = node.reward
    while node is not None:
        node.n += 1
        node.q = ((node.n - 1) / node.n) * node.q + 1 / node.n * r
        node = node.parent

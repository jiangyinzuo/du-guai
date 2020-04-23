# -*- coding: utf-8 -*-
from card import CARD_G0, CARD_G1
from ml import LandlordClassifier
from application.shuffle import shuffle

import numpy as np


def test_call_landlord():
    landlord_clf = LandlordClassifier()
    landlord_clf.fit(shuffle(17))

    ghosts: np.ndarray = landlord_clf.cards[-2:]
    if landlord_clf.ghost == 3:
        assert CARD_G0 in ghosts and CARD_G1 in ghosts
    elif landlord_clf.ghost == 2:
        assert CARD_G1 in ghosts and CARD_G0 not in ghosts
    elif landlord_clf.ghost == 1:
        assert CARD_G0 in ghosts and CARD_G1 not in ghosts
    else:
        assert CARD_G0 not in ghosts and CARD_G1 not in ghosts

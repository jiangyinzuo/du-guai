# -*- coding: utf-8 -*-

import numpy as np

if __name__ == '__main__':
    arr = np.zeros((5, 8))
    arr[2, 4] = 2
    print(arr)
    np.save('datas.npy', arr)

    res = np.load('./datas.npy', allow_pickle=True)
    print(res)

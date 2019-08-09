# coding=utf-8

import util
import random
import numpy as np


def test_shuffle():
    a = np.random.randint(0, 10, (4, 3))
    b = np.random.randint(0, 10, 4)
    c = np.c_[a, b]
    np.random.shuffle(c)
    d = c[:, :a.shape[1]]
    e = c[:, a.shape[1]:]
    print(a)
    print('-------------------------------')
    print(b)
    print('-------------------------------')
    print(c)
    print('-------------------------------')
    print(d)
    print('-------------------------------')
    print(e)

def test_dataframe():
    import pandas as pd

    a = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
    b = {}
    sum = 0
    for k, v in a.items():
        sum += v
    for k, v in a.items():
        b[k] = (1-v/sum)/(len(a)-1)
    print(b)
    df = pd.DataFrame(b, index=[0])
    print(df)

def test_time():
    import time

    a = 'model/' + str(int(time.time()))
    print(a)


if __name__ == '__main__':
    # test_shuffle()
    # test_dataframe()
    test_time()

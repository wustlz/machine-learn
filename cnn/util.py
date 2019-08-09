# coding=utf-8

import logging, os
import numpy as np


def shuffle_set(x_data, y_label):
    """
    随机打散数据
    :param x_data: X
    :param y_label: Y
    :return:
    """
    x = np.array(x_data)
    y = np.array(y_label)
    x_y = np.c_[x, y]
    np.random.shuffle(x_y)
    x_data = x_y[:, :x.shape[1]]
    y_label = x_y[:, x.shape[1]:]

    return x_data, y_label


def get_batch(data, label, batch_size, now_batch, total_batch):
    """
    按照指定batch大小返回数据
    :param data: 数据
    :param label: 标签
    :param batch_size: batch大小
    :param now_batch:
    :param total_batch:
    :return:
    """
    if now_batch < total_batch - 1:
        data_batch = data[now_batch * batch_size:(now_batch + 1) * batch_size]
        label_batch = label[now_batch * batch_size:(now_batch + 1) * batch_size]
    else:
        data_batch = data[now_batch * batch_size:]
        label_batch = label[now_batch * batch_size:]
    return data_batch, label_batch


def get_logger(filename):
    """
    logger定义
    :param filename:
    :return:
    """
    file_dir = os.path.split(filename)[0]
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

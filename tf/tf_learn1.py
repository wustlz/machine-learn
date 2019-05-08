# encoding=utf-8

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告


def base_opration():
    """
    TensorFlow的基本操作
    """
    x = tf.constant([1, 2])
    y = tf.constant([3, 4])
    z = tf.add(x, y)
    a = tf.Variable(tf.random_normal([5, 5, 1, 32]))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(a)
        print(result)


if __name__ == "__main__":
    base_opration()
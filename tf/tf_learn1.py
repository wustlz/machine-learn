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


def get_value():
    a = scope_op()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(a)
        print(result)


def scope_op():
    with tf.variable_scope('test_scope'):
        a = tf.constant([1, 2])
        return a

def test_reshape():
    a = tf.get_variable('a', [3, 4, 4, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a_shape = a.get_shape().as_list()
        print(a_shape)
        b = tf.reshape(a, shape=[3*128])
        b_shape = b.get_shape()
        print(b_shape[0])

def test_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("../data/mnist/", one_hot=True)
    print(mnist.train.num_examples)
    batch_x, batch_y = mnist.train.next_batch(100)
    print(batch_x)
    print('---------------------')
    print(batch_y)

if __name__ == "__main__":
    # base_opration()
    # get_value()
    # test_reshape()
    test_mnist()

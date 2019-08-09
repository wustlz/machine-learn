# _*_ coding: utf-8 _*_
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist_inference.py中定义的常量和前向传播的函数
from model_tf import LeNet5

def main(argv=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    mnist = input_data.read_data_sets("../data/mnist/", one_hot=True)
    leNet_model = LeNet5(input_size=[28, 28, 1], out_class=10, config=config)

    input_x = mnist.train.images
    target_y = mnist.train.labels
    leNet_model.train(input_x=input_x, target_y=target_y)

if __name__ == '__main__':
    tf.app.run()
# _*_ coding: utf-8 _*_
import tensorflow as tf
import numpy as np
import os, time
import util

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0

class LeNet5(object):
    """
    CNN中的经典模型LeNet5的TensorFlow实现
    """

    def __init__(self, input_size, out_class, conv_size=[5, 5], conv_deep=[32, 64], conv_strides=[1, 1], pool_size=[2, 2],
                 pool_strides=[2, 2], fc_size=[512], learning_rate_base=0.1, learning_rate_decay=0.99, train_steps=5000,
                 moving_average_decay=0.99, regularization_rate=0.0001, batch_size=100, config=None):
        """
        配置神经网络的参数
        :param input_size: 输入维数 [n, m, deep]
        :param out_class: 输出类别数
        :param conv_size: 卷积核大小 [kernel_size, ...]
        :param conv_deep: 卷积核深度 [kernel_deep, ...]
        :param conv_strides: 卷积核步长 [kernel_strides, ...]
        :param pool_size: 池化过滤器大小 [pool_size, ...]
        :param pool_strides: 池化过滤器移动步长 [pool_size, ...]
        :param fc_size: 全连接层节点数 [fc_node_num, ...]
        :param learning_rate_base: 基础学习率
        :param learning_rate_decay: 学习率衰减因子
        :param train_steps: 最大训练步长
        :param moving_average_decay: 损失函数计算(梯度下降法)因子
        :param regularization_rate: 正则化因子，全连接层使用，表示模型负责度
        :param batch_size: 单次训练batch大小
        :param config: TensorFlow运行config配置，主要是启动GPU
        """

        cur_time = str(int(time.time()))

        # 输入向量
        self.input_n = input_size[0]
        self.input_m = input_size[1]
        self.input_deep = input_size[2]

        # 输出类别数
        self.num_labels = out_class
        self.model_path = './model/' + cur_time + '/'
        self.model_name = 'leNet.ckpt'

        self.batch_size = batch_size

        # 卷积层相关参数
        self.conv1_deep = conv_deep[0]
        self.conv2_deep = conv_deep[1]
        self.conv_layer_name = ['conv1', 'conv2']
        self.conv_param = {
            self.conv_layer_name[0]: {
                'conv_size': conv_size[0],
                'conv_deep': self.conv1_deep,
                'pre_deep': self.input_deep,
                'conv_strides': conv_strides[0],
                'pool_size': pool_size[0],
                'pool_strides': pool_strides[0]
            },
            self.conv_layer_name[1]: {
                'conv_size': conv_size[0],
                'conv_deep': self.conv2_deep,
                'pre_deep': self.conv1_deep,
                'conv_strides': conv_strides[1],
                'pool_size': pool_size[1],
                'pool_strides': pool_strides[1]
            }
        }

        # 全连接层的节点个数
        self.fc1_size = fc_size[0]

        # 超参数
        self.learning_rate_base = learning_rate_base
        self.learning_rate_decay = learning_rate_decay
        self.train_steps = train_steps
        self.moving_average_decay = moving_average_decay

        self.regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
        self.global_step = tf.Variable(0, trainable=False)

        self.input_tensor = tf.placeholder(tf.float32, [self.batch_size, self.input_n, self.input_m, self.input_deep],
                                           name='x-input')
        self.output_label = tf.placeholder(tf.float32, [None, self.num_labels], name='y-input')

        self.log_path = './logs/tf_leNet5_' + cur_time + '.log'
        self.logger = util.get_logger(self.log_path)
        self.config = config

    def train(self, input_x, target_y):

        total_sample = len(target_y)

        # leNet5 model计算
        # self._model(self.train)   # 通过train控制是否进行dropout
        self._model()

        # 损失值表示
        self._loss()

        # 准确率计算
        self._evaluate()

        # Optimizer定义
        self._optimizer(total_sample)

        # 开始训练
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            # 保存日志
            # file_writer = tf.summary.FileWriter(self.log_path, sess.graph)

            # 打散原数据
            X, Y = util.shuffle_set(input_x, target_y)
            now_batch = 0
            total_batch = total_sample / self.batch_size

            for i in range(self.train_steps):
                # 若当前batch已经超过实际数量，则需要重新打散开始
                if now_batch > total_batch:
                    X, Y = util.shuffle_set(input_x, target_y)
                    now_batch = 0
                # 获取batch数据
                xs, ys = util.get_batch(X, Y, self.batch_size, now_batch, total_batch)
                xs = np.reshape(xs, [self.batch_size, self.input_n, self.input_m, self.input_deep])
                _, loss_value, step, summary = sess.run([self.train_op, self.loss, self.global_step, self.train_summary_op],
                                               feed_dict={self.input_tensor: xs, self.output_label: ys})
                # file_writer.add_summary(summary, step)
                if i % 1000 == 0 or step == self.train_steps-1:
                    self.logger.info("After %d training steps, loss on training batch is %g" % (step, loss_value))
                    saver.save(sess, os.path.join(self.model_path, self.model_name), global_step=self.global_step)

    def _optimizer(self, sample_size):
        """
        优化器定义
        :param sample_size: 总样本数，学习率衰减使用
        :return:
        """
        learning_rate = tf.train.exponential_decay(self.learning_rate_base,
                                                   global_step=self.global_step,
                                                   decay_steps=sample_size / self.batch_size,
                                                   decay_rate=self.learning_rate_decay)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss, global_step=self.global_step)
        with tf.control_dependencies([train_step, self.variable_average_op]):
            self.train_op = tf.no_op(name='train')

    def _loss(self):
        """
        损失函数定义
        :return:
        """
        variable_average = tf.train.ExponentialMovingAverage(self.moving_average_decay, self.global_step)
        self.variable_average_op = variable_average.apply(tf.trainable_variables())
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.output_label, 1), logits=self.logit)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        self.loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    def _evaluate(self):
        """
        准确率计算
        :return:
        """
        prediction = tf.nn.softmax(self.logit)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.output_label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        training_accuracy = tf.summary.scalar("accuracy", self.accuracy)
        cost = tf.summary.scalar("cost", self.loss)
        self.train_summary_op = tf.summary.merge([cost, training_accuracy])

    def _model(self):
        """
        LeNet5模型定义
        :return:
        """

        # 第1层卷积池化
        pool1 = self._convolution_pool_layer(self.conv_layer_name[0], self.input_tensor)

        # 第2层卷积池化
        pool2 = self._convolution_pool_layer(self.conv_layer_name[1], pool1)

        # 需要先计算输入节点数
        pool2_shape = pool2.get_shape().as_list()
        nodes = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]

        fc_input_vector = tf.reshape(pool2, [pool2_shape[0], nodes])

        # 第3层全连接层
        fc1 = self._full_conn_layer('fc1', fc_input_vector, nodes, self.fc1_size)
        # 通过dropout避免过拟合
        # if train:
        fc1 = tf.nn.dropout(fc1, 0.5)

        # 第4层全连接层
        self.logit = self._full_conn_layer('fc2', fc1, self.fc1_size, self.num_labels)

    def _convolution_pool_layer(self, conv_layer, input_tensor):
        """
        卷积池化层计算
        :param input_tensor: 输出张量
        :param conv_layer: 卷积池化层名称
        :return:
        """
        with tf.variable_scope('layer-' + conv_layer):
            conv_param = self.conv_param.get(conv_layer)
            conv_size = conv_param.get('conv_size')
            conv_strides = conv_param.get('conv_strides')
            pool_size = conv_param.get('pool_size')
            pool_strides = conv_param.get('pool_strides')

            conv_weights = tf.get_variable('weight',
                                           [conv_size, conv_size, conv_param.get('pre_deep'), conv_param.get('conv_deep')],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv_biases = tf.get_variable('bias', [conv_param.get('conv_deep')],
                                           initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(input_tensor, conv_weights, strides=[1, conv_strides, conv_strides, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))

            pool = tf.nn.max_pool(relu, ksize=[1, pool_size, pool_size, 1],
                                   strides=[1, pool_strides, pool_strides, 1], padding='SAME')
            return pool

    def _full_conn_layer(self, fc_layer, input_vector, fc_in_size, fc_out_size):
        """
        全连接层
        :param fc_layer: 全连接层名称
        :param input_vector: 输入向量
        :param fc_in_size: 输入维数
        :param fc_out_size: 输出节点数
        :return:
        """
        with tf.variable_scope('layer-' + fc_layer):
            fc_weights = tf.get_variable('weight', [fc_in_size, fc_out_size],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            if self.regularizer is not None:
                tf.add_to_collection('losses', self.regularizer(fc_weights))
            fc_biases = tf.get_variable('bias', [fc_out_size], initializer=tf.constant_initializer(0.0))
            return tf.matmul(input_vector, fc_weights) + fc_biases

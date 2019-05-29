# -*- coding:utf-8 -*-
import tensorflow as tf


class CNN:

    def __init__(self, save_or_load_path=None, trainable=True, learning_rate = 0.00002,timestep=9,road=189,predstep=1):
        self.trainable = trainable
        self.learning_rate = learning_rate
        self.road = road
        self.input_size = timestep * road
        self.output_size = predstep * road
        self.bottom = tf.placeholder(tf.float32, shape=[None, self.input_size], name='input')  # 25*2*6
        self.target = tf.placeholder(tf.float32, shape=[None, self.output_size], name='target')
        self.timestep = timestep

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return initial

    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def conv1d(self,x, W):
        return tf.nn.conv1d(x, W, stride=2, padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def build_CNN(self, ):

        # conv first
        bottom = tf.reshape(self.bottom, [-1, self.road, self.timestep, 1])
        W_conv1 = self.weight_variable([3, 3, 1, 64])
        b_conv1 = self.bias_variable([64])
        h_conv1 = tf.nn.elu(self.conv2d(bottom, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_flat3 = tf.reshape(h_pool1, [-1, 95 * 5 * 64])
        W_fc2 = self.weight_variable([95 * 5  * 64, 1200])
        b_fc2 = self.bias_variable([1200])
        h = tf.nn.elu(tf.matmul(h_flat3, W_fc2) + b_fc2)
        # h_flat3 = tf.reshape(h_pool3, [-1, 400])
        W_fc2 = self.weight_variable([1200, self.output_size])
        b_fc2 = self.bias_variable([self.output_size])
        self.predict = tf.nn.elu(tf.matmul(h, W_fc2) + b_fc2)


        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = 0.0002 #tf.train.exponential_decay(0.001,  global_step,  500, 0.9,staircase=True)
        self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.predict))
        self.accuracy = 1. - tf.reduce_mean(abs(self.target-self.predict)/self.target)
        self.trainop = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)
        # self.trainop = tf.train.RMSPropOptimizer(self.learning_rate, 0.99, 0.0, 1e-6).minimize(self.loss)
        return self.predict



class CNN15:
    def __init__(self, save_or_load_path=None, trainable=True, learning_rate = 0.00002,timestep=9,road=189,predstep=3):
        self.trainable = trainable
        self.learning_rate = learning_rate
        self.road = road
        self.input_size = timestep * road
        self.output_size = predstep * road
        self.bottom = tf.placeholder(tf.float32, shape=[None, self.input_size], name='input')  # 25*2*6
        self.target = tf.placeholder(tf.float32, shape=[None, self.output_size], name='target')
        self.timestep = timestep

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return initial

    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def conv1d(self,x, W):
        return tf.nn.conv1d(x, W, stride=2, padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def build_CNN(self, ):

        # conv first
        bottom = tf.reshape(self.bottom, [-1, self.road, self.timestep, 1])
        W_conv1 = self.weight_variable([3, 3, 1, 64])
        b_conv1 = self.bias_variable([64])
        h_conv1 = tf.nn.elu(self.conv2d(bottom, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_flat3 = tf.reshape(h_pool1, [-1, 95 * 5 * 64])
        W_fc2 = self.weight_variable([95 * 5  * 64, 1200])
        b_fc2 = self.bias_variable([1200])
        h = tf.nn.elu(tf.matmul(h_flat3, W_fc2) + b_fc2)
        # h_flat3 = tf.reshape(h_pool3, [-1, 400])
        W_fc2 = self.weight_variable([1200, self.output_size])
        b_fc2 = self.bias_variable([self.output_size])
        self.predict = tf.nn.elu(tf.matmul(h, W_fc2) + b_fc2)


        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = 0.0002 #tf.train.exponential_decay(0.001,  global_step,  500, 0.9,staircase=True)
        self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.predict))
        self.accuracy = 1. - tf.reduce_mean(abs(self.target-self.predict)/self.target)
        self.trainop = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)
        # self.trainop = tf.train.RMSPropOptimizer(self.learning_rate, 0.99, 0.0, 1e-6).minimize(self.loss)
        return self.predict




class CNN30:
    def __init__(self, save_or_load_path=None, trainable=True, learning_rate=0.00002,timestep=9,road=189,predstep=6):
        self.trainable = trainable
        self.learning_rate = learning_rate
        self.road = road
        self.input_size = timestep * road
        self.output_size = predstep * road
        self.bottom = tf.placeholder(tf.float32, shape=[None, self.input_size], name='input')  # 25*2*6
        self.target = tf.placeholder(tf.float32, shape=[None, self.output_size], name='target')
        self.timestep = timestep

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return initial

    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def conv1d(self,x, W):
        return tf.nn.conv1d(x, W, stride=2, padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    def build_CNN(self, ):
        # conv first
        bottom = tf.reshape(self.bottom, [-1, self.road, self.timestep, 1])
        W_conv1 = self.weight_variable([3, 3, 1, 64])
        b_conv1 = self.bias_variable([64])
        h_conv1 = tf.nn.elu(self.conv2d(bottom, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_flat3 = tf.reshape(h_pool1, [-1, 95 * 5 * 64])
        W_fc2 = self.weight_variable([95 * 5 * 64, 1200])
        b_fc2 = self.bias_variable([1200])
        h = tf.nn.elu(tf.matmul(h_flat3, W_fc2) + b_fc2)
        # h_flat3 = tf.reshape(h_pool3, [-1, 400])
        W_fc2 = self.weight_variable([1200, self.output_size])
        b_fc2 = self.bias_variable([self.output_size])
        self.predict = tf.nn.elu(tf.matmul(h, W_fc2) + b_fc2)

        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = 0.0002  # tf.train.exponential_decay(0.001,  global_step,  500, 0.9,staircase=True)
        self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.predict))
        self.accuracy = 1. - tf.reduce_mean(abs(self.target - self.predict) / self.target)
        self.trainop = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)
        # self.trainop = tf.train.RMSPropOptimizer(self.learning_rate, 0.99, 0.0, 1e-6).minimize(self.loss)
        return self.predict
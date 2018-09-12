'''
Input: state s
Output: action a, as a one-hot vector
'''


import tensorflow as tf
import numpy as np
import random

#–––––––––––––––––––––––––––––Hyperparameters––––––––––––––––––––––––––#
# File names
csv_file = 'data_set.csv'
weights_file = 'weights/model.ckpt'

# Input
map_width = 100
map_height = 100
num_chan = 7

# Layer 1
conv1_out_num = 32
pool1_width = map_width / 2
pool1_height = map_height / 2

# Layer 2
conv2_out_num = 64
pool2_width = pool1_width / 2
pool2_height = pool1_height / 2

# Layer 3
conv3_out_num = 64
pool3_width = pool2_width / 2
pool3_height = pool2_height / 2

# Output
num_acts = 5

# Reinforcement learning parameters
init_epsilon = 1.0
fin_epsilon = 0.05
explore = 500


def decision_maker(s_in):
    epsilon = init_epsilon
    a = np.zeros(num_acts)
    if random.random() <= epsilon:
        a_index = random.randrange(num_acts)
    else:
        # TODO: Is this right?
        # scale down epsilon
        if epsilon > fin_epsilon:
            epsilon -= (init_epsilon - fin_epsilon) / explore

        #––––––––––––––––––––––––––––---load weight–––––––––––––––––––––––––––––#
        # Load weight
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, weights_file)

        #–––––––––––––––––––––––––––––––––––CNN–––––––––––––––––––––––––––––––––#

        # Placeholders for inputs
        s = tf.placeholder(tf.float32, shape=[None, map_width, map_height, num_chan])
        a = tf.placeholder(tf.float32, shape=[None, num_acts])


        # Convolutional layer 1, out:map_width x map_height x conv1_num_chan
        with tf.name_scope('conv1') as scope:
            # Kernel 3 x 3
            w_conv1 = tf.truncated_normal([3, 3, num_chan, conv1_out_num], stddev=0.1)
            b_conv1 = tf.constant(0.1, [conv1_out_num])
            h_conv1 = tf.nn.relu(tf.conv2d(s, w_conv1) + b_conv1)

        # Pooling layer 1, out: pool1_width x pool1_height x conv1_out_num
        with tf.name_scope('pool1') as scope:
            h_pool1 = tf.nn.max_pool(h_conv1,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME')

        # Convolutional layer 2, out:pool1_width x pool1_height x conv2_out_num
        with tf.name_scope('conv2') as scope:
            # Kernel 3 x 3
            w_conv2 = tf.truncated_normal([3, 3, num_chan, conv2_out_num], stddev=0.1)
            b_conv2 = tf.constant(0.1, [conv2_out_num])
            h_conv2 = tf.nn.relu(tf.conv2d(h_pool1, w_conv2) + b_conv2)

        # Pooling layer 2, out: pool2_width x pool2_height x conv2_out_num
        with tf.name_scope('pool2') as scope:
            h_pool2 = tf.nn.max_pool(h_conv2,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME')

        # Convolutional layer 3, out:pool2_width x pool2_height x conv3_out_num
        with tf.name_scope('conv3') as scope:
            # Kernel 3 x 3
            w_conv3 = tf.truncated_normal([3, 3, num_chan, conv3_out_num], stddev=0.1)
            b_conv3 = tf.constant(0.1, [conv1_out_num])
            h_conv3 = tf.nn.relu(tf.conv2d(h_pool2, w_conv3) + b_conv3)

        # Pooling layer 3, out: pool3_width x pool3_height x conv3_out_num
        with tf.name_scope('pool3') as scope:
            h_pool3 = tf.nn.max_pool(h_conv3,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME')

        # Fully connected layer
        # pred_act is the predicted act
        with tf.name_scope('full1') as scope:
            w_full = tf.truncated_normal([pool3_width * pool3_height * conv3_out_num, num_acts])
            b_full = tf.constant(0.1, [num_acts])
            q_s = tf.matmul(h_pool3, w_full) + b_full

        # Loss function: Greater the reward, smaller the loss
        q = q_s.eval(feed_dict={s: s_in})
        a_index = np.argmax(q)

    a[a_index] = 1
    return a


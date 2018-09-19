'''
CNN for training the bot

Aim: Learns Q(s, a)

Input: TODO: Update the specification

Output: Q(s, a), approximating Q*


DQN:
s: state, numpy array, [input data map_width x map_height x num_chan]
a: action, numpy array, [num_acts]
r: reward, float
q_s: Q(s, a), numpy array [num_acts] (for all a')
q_s_a: Q (s, a), float
q_: Q*(s', a'), numpy array [num_acts] (for all a)


'''


import tensorflow as tf
import numpy as np
import random
import os
import pickle
from storage import TrainingStorage

# File names
pickle_file = 'training.p'
pickle_debug = 'dummy_data_big.p'
weights_file = 'model.ckpt'
weights_dir = 'tools/weights'


# Input
input_size = 16
num_chan = 6

# Layer 1
# Map size must be multiples of 8
conv1_out_num = 32
pool1_size = int(input_size / 2)

# Layer 2
conv2_out_num = 64
pool2_size = int(pool1_size / 2)

# Layer 3
conv3_out_num = 64
pool3_size = int(pool2_size / 2)

# Output
num_acts = 5

# Reinforcement learning parameters
gamma = 0.9
batch_size = 5
l_rate = 0.1
epoch = 10


def process_data(data):
    data_s = []

    for i, (s, a, r) in enumerate(data):
        if i + 1 < len(data):
            data_s.append((s, a, r, data[i + 1][0]))  # data[i+1][0] is s_

    # Create minibatches
    minibatches = []
    batches = []

    random.shuffle(data_s)

    for i in range(0, len(data) - batch_size, batch_size):
        minibatches.append(data_s[i:i + batch_size])

    for minibatch in minibatches:
        # Separately store minibatches
        s_batch = np.array([single_data[0] for single_data in minibatch])
        a_batch = np.array([single_data[1] for single_data in minibatch])
        r_batch = np.array([single_data[2] for single_data in minibatch])
        s_batch_ = np.array([single_data[3] for single_data in minibatch])

        s_batch = np.reshape(s_batch, (batch_size, input_size, input_size, num_chan))
        a_batch = np.reshape(a_batch, (batch_size, num_acts))
        r_batch = np.reshape(r_batch, (batch_size, 1))
        s_batch_ = np.reshape(s_batch_, (batch_size, input_size, input_size, num_chan))

        batches.append((s_batch, a_batch, r_batch, s_batch_))
    return batches


def get_data():

    # TODO: need to handle exception
    trainingstorage = TrainingStorage()
    if os.path.exists(pickle_file):
        all_data = list(trainingstorage.items())
    else:
        all_data = pickle.load(open(pickle_debug, "rb"))

    # Sort data according to label of ants
    sorted_data = []
    labels = []
    for (s, a, r, label, turn) in all_data:

        # Check for new ants
        if label not in labels:
            labels.append(label)
            ant_data = []
            sorted_data.append(ant_data)

        for index in range(len(labels)):
            if label == index:
                sorted_data[label].append((s, a, r))

    # Preprocess data and create batches
    batches = []
    for ant_data in sorted_data:
        ant_batches = process_data(ant_data)
        batches.append(ant_batches)

    return batches


def create_network():
    # Placeholders for s
    s = tf.placeholder(tf.float32, shape=[None, input_size, input_size, num_chan])

    def conv_layer(in_data, in_chan, out_chan, name):
        w_conv = tf.Variable(tf.truncated_normal([3, 3, in_chan, out_chan], stddev=0.1), name='w_conv' + name)
        b_conv = tf.Variable(tf.constant(0.1, shape=[out_chan]), name='b_conv' + name)
        return tf.nn.relu(tf.nn.conv2d(in_data, w_conv, strides=[1,1,1,1], padding='SAME') + b_conv), w_conv, b_conv

    def pooling_layer(conv):
        h_pool = tf.nn.max_pool(conv,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')
        return h_pool

    def full_layer(in_data, num_neuron, num_out):
        w_full = tf.Variable(tf.truncated_normal([num_neuron, num_out]), name='w_full')
        b_full = tf.Variable(tf.constant(0.1, shape=[num_out]), name='b_full')
        return tf.matmul(in_data, w_full) + b_full, w_full, b_full

    # Convolutional layer 1, out:map_width x map_height x conv1_num_chan
    conv1, w_conv1, b_conv1 = conv_layer(s, num_chan, conv1_out_num, '1')

    # Pooling layer 1, out: pool1_width x pool1_height x conv1_out_num
    pool1 = pooling_layer(conv1)

    # Convolutional layer 2, out:pool1_width x pool1_height x conv2_out_num
    conv2, w_conv2, b_conv2 = conv_layer(pool1, conv1_out_num, conv2_out_num, '2')

    # Pooling layer 2, out: pool2_width x pool2_height x conv2_out_num
    pool2 = pooling_layer(conv2)

    # Convolutional layer 3, out:pool2_width x pool2_height x conv3_out_num
    conv3, w_conv3, b_conv3 = conv_layer(pool2, conv2_out_num, conv3_out_num, '3')

    # Pooling layer 3, out: pool3_width x pool3_height x conv3_out_num
    pool3 = pooling_layer(conv3)

    # Fully connected layer
    q_s, w_full, b_full = full_layer(tf.layers.flatten(pool3), pool3_size * pool3_size * conv3_out_num, num_acts)

    variables = (w_conv1, w_conv2, w_conv3, b_conv1, b_conv2, b_conv3, w_full, b_full)
    return q_s, s, variables


def train_network(q_s, s, sess, batches, variables):
    # Placeholders
    a = tf.placeholder(tf.float32, shape=[None, num_acts])
    y = tf.placeholder(tf.float32, shape=[None])

    # Loss function
    q_s_a = tf.reduce_sum(tf.multiply(q_s, a))
    loss = tf.reduce_mean(tf.square(y - q_s_a))
    train_step = tf.train.AdamOptimizer(l_rate).minimize(loss)


    w_conv1, w_conv2, w_conv3, b_conv1, b_conv2, b_conv3, w_full, b_full = variables

    # initialize
    sess.run(tf.global_variables_initializer())
    print('initialized')

    if os.path.exists(weights_dir):
        # Load weight
        saver = tf.train.Saver({'w_conv1': w_conv1,
                                'w_conv2': w_conv2,
                                'w_conv3': w_conv3,
                                'b-conv1': b_conv1,
                                'b_conv2': b_conv2,
                                'b_conv3': b_conv3,
                                'w_full': w_full,
                                'b_full': b_full})
        path = os.path.join(weights_dir, weights_file)
        saver.restore(sess, path)
        print('Weights loaded')

    # Training
    for i in range(epoch):
        for batch in batches:
            (s_batch, a_batch, r_batch, s_batch_) = batch
            y_batch = []

            q_s_a_t = q_s.eval(feed_dict={s: s_batch_})

            for j in range(batch_size):
                y_batch.append(r_batch[j][0] + gamma * np.max(q_s_a_t))

            train_step.run(feed_dict={y: y_batch, a: a_batch, s: s_batch_})

        loss_val = sess.run(loss, feed_dict={y: y_batch, a: a_batch, s: s_batch_})
        print('Epoch: %d, Loss: %f' % (i, loss_val))
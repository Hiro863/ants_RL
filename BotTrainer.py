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
from pickle import Unpickler
from storage import TrainingStorage

# File names
pickle_file = 'training.p'
pickle_file_observe = 'observing.p'
pickle_debug = 'debug_data.p'
pickle_test = 'debug_test.p'
weights_file = 'model.ckpt'
weights_dir = 'weights'
target_dir = 'target_weights'
target_weights_file = 'model.ckpt'


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

# Fully Connected
fc_neuron = 256

# Output
num_acts = 5

# Reinforcement learning parameters
gamma = 0.99
batch_size = 30
l_rate = 1e-6
epoch = 1


def add_next_state(data):
    # converts (s, a, r) to (s, a, r, s')
    data_s = []

    for i, (s, a, r) in enumerate(data):
        if i + 1 < len(data):
            data_s.append((s, a, r, data[i + 1][0]))
        else:  # TODO suboptimal solution
            data_s.append((s, a, r, s))

    return data_s


def create_batches(data):

    # Create minibatches
    minibatches = []
    batches = []

    # shaffling before creating batches
    random.shuffle(data)

    # dividing data into minibatches
    for i in range(0, len(data) - batch_size, batch_size):
        minibatches.append(data[i:i + batch_size])

    # reshape each elements to right shape in np.array form
    for minibatch in minibatches:

        # empty arrays
        s_batch = np.empty((0, input_size, input_size, num_chan))
        a_batch = np.empty((0, num_acts))
        r_batch = np.empty((0, 1))
        s_batch_ = np.empty((0, input_size, input_size, num_chan))

        # process for each tuple
        for single_data in minibatch:
            s, a, r, s_ = single_data

            # stack channel by channel, s
            new_s = np.empty((input_size, input_size, 0))
            for channel in s:
                channel = np.reshape(channel, (input_size, input_size, 1))
                new_s = np.append(new_s, channel, axis=2)

            # simply turn into arrays
            new_a = np.array(a)
            new_r = np.array(r)

            # stack channel by channel, s'
            new_s_ = np.empty((input_size, input_size, 0))
            for channel in s_:
                channel = np.reshape(channel, (input_size, input_size, 1))
                new_s_ = np.append(new_s_, channel, axis=2)

            # reshape, first dimension corresponds to batch size
            new_s = np.reshape(new_s, (1, input_size, input_size, num_chan))
            new_a = np.reshape(new_a, (1, num_acts))
            new_r = np.reshape(new_r, (1, 1))
            new_s_ = np.reshape(new_s_, (1, input_size, input_size, num_chan))

            # append single data to the batch
            s_batch = np.append(s_batch, new_s, axis=0)
            a_batch = np.append(a_batch, new_a, axis=0)
            r_batch = np.append(r_batch, new_r, axis=0)
            s_batch_ = np.append(s_batch_, new_s_, axis=0)

        # tuple of each batches
        batches.append((s_batch, a_batch, r_batch, s_batch_))
    return batches


def get_data(session_mode):

    # Loading data from pickle file
    # TODO: clean this bit up
    if session_mode == 'observing':
        print('loading observe data...')
        trainingstorage = TrainingStorage(file=pickle_file_observe)
        if os.path.exists(pickle_file_observe):
            all_data = list(trainingstorage.items())
            print('observing')
        else:
            return
    elif session_mode == 'debug':
        print('loading debug data')
        with open(pickle_debug, 'rb') as f:
            unpickler = Unpickler(f)
            try:
                batches = unpickler.load()
                return batches
            except EOFError:
                pass
    elif session_mode == 'test':
        print('loading test data')
        with open(pickle_test, 'rb') as f:
            unpickler = Unpickler(f)
            try:
                batches = unpickler.load()
                return batches
            except EOFError:
                pass
    else:
        print('loading train data')
        trainingstorage = TrainingStorage()
        if os.path.exists(pickle_file):
            all_data = list(trainingstorage.items())
        else:
            return

    # Sort data according to label of ants
    sorted_data = []
    labels = []
    for (s, a, r, label, turn) in all_data:

        # Check for new ants
        if label not in labels:
            labels.append(label)
            ant_data = []
            sorted_data.append(ant_data)

        # append ants to different sublists according to the label
        for index in range(len(labels)):
            if label == index:
                sorted_data[label].append((s, a, r))



    # add s' to (s, a, r)
    data_added = []
    for ant_data in sorted_data:
        data_added.append(add_next_state(ant_data))

    # concatenate all tuples
    concatenated = []
    for ant_data_added in data_added:
        for single_data in ant_data_added:
            concatenated.append(single_data)

    # create minibatches
    batches = create_batches(concatenated)
    return batches


def create_network():
    # Placeholders for s
    s = tf.placeholder(tf.float32, shape=[None, input_size, input_size, num_chan])

    def conv_layer(in_data, in_chan, out_chan, name):
        w_conv = tf.Variable(tf.truncated_normal([3, 3, in_chan, out_chan], stddev=0.1), name='w_conv' + name)
        b_conv = tf.Variable(tf.constant(0.1, shape=[out_chan]), name='b_conv' + name)
        return tf.nn.relu(tf.nn.conv2d(in_data, w_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv), w_conv, b_conv

    def pooling_layer(conv):
        h_pool = tf.nn.max_pool(conv,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')
        return h_pool

    def full_layer(in_data, num_neuron, num_out, name):
        w_full = tf.Variable(tf.truncated_normal([num_neuron, num_out]), name='w_full' + name)
        b_full = tf.Variable(tf.constant(0.1, shape=[num_out]), name='b_full' + name)
        return tf.matmul(in_data, w_full) + b_full

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
    fc = tf.nn.relu(full_layer(tf.layers.flatten(pool3), pool3_size * pool3_size * conv3_out_num, fc_neuron, '1'))

    # Dropout layer
    keep_prob = tf.placeholder(tf.float32)
    fc_drop = tf.nn.dropout(fc, keep_prob)

    # Output layer
    q_s = full_layer(fc_drop, fc_neuron, num_acts, '2')

    return q_s, s, keep_prob


def train_network(q_s, s, sess, batches, target_batches, keep_prob, session_mode):
    # Placeholders
    a = tf.placeholder(tf.float32, shape=[None, num_acts])
    y = tf.placeholder(tf.float32, shape=[None, 1])

    # Loss function
    q_s_a = tf.reduce_sum(tf.multiply(q_s, a), keepdims=True)
    #loss = tf.reduce_mean(tf.square(y - q_s_a))
    loss = tf.losses.huber_loss(labels=y, predictions=q_s_a)
    train_step = tf.train.AdamOptimizer(l_rate).minimize(loss)

    # initialize
    if os.path.exists(weights_dir):
        saver = tf.train.Saver()
        weights_path = os.path.join(weights_dir, weights_file)
        saver.restore(sess, weights_path)
        print('weights loaded')
    else:
        sess.run(tf.global_variables_initializer())
        print('initialized')

    # Training
    last_loss = 0
    for i in range(epoch):
        for batch, target_q_batch in zip(batches, target_batches):
            (s_batch, a_batch, r_batch, s_batch_) = batch
            y_batch = []

            # get Q value for the next state
            # TODO: get q_s value from target Q
            #q_s_a_t = q_s.eval(feed_dict={s: s_batch_, keep_prob: 1.0})

            # calculate target value using Bellman equation
            for j in range(batch_size):
                #y_batch.append(r_batch[j] + gamma * np.max(q_s_a_t[j]))
                y_batch.append(r_batch[j] + gamma * np.max(target_q_batch[j]))

            # train the neural network
            train_step.run(feed_dict={y: y_batch, a: a_batch, s: s_batch, keep_prob: 1.0})

        # print the loss value
        loss_val = sess.run(loss, feed_dict={y: y_batch, a: a_batch, s: s_batch, keep_prob: 1.0})

        print('Epoch: %d, Loss: %f' % (i, loss_val))

        last_loss = loss_val
    return last_loss


class TargetQ:
    def __init__(self):
        # Define Session
        self.target_sess = tf.InteractiveSession()

        # Create network
        self.q_s, self.s, self.keep_prob = create_network()

        # load weights or initialize
        path = os.path.join(target_dir, target_weights_file)
        if os.path.exists(target_dir):
            saver = tf.train.Saver()
            saver.restore(self.target_sess, path)
            print('target weights loaded')
        else:
            self.target_sess.run(tf.global_variables_initializer())
            print('target initialized')

    def get_target_q(self, batches):
        # get the target value
        target_batch = []

        for batch in batches:
            q_s_a_t_batch = []
            s_batch, a_batch, r_batch, s_batch_ = batch

            for s_ in s_batch_:
                s_ = np.reshape(s_, (1, input_size, input_size, num_chan))
                q_s_a_t = self.target_sess.run(self.q_s, feed_dict={self.s: s_, self.keep_prob: 1.0})
                q_s_a_t = np.reshape(q_s_a_t, num_acts)
                q_s_a_t_batch.append(q_s_a_t)

            target_batch.append(q_s_a_t_batch)
        return target_batch





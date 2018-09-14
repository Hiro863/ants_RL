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
import os.path
import pickle

#–––––––––––––––––––––––––––––Hyperparameters––––––––––––––––––––––––––#
# File names
pickle_file = 'dummy_data.p'
weights_file = 'weights/model.ckpt'

# Input
map_width = 80
map_height = 80
num_chan = 7

# Layer 1
# TODO: Map size might need to be adjusted
conv1_out_num = 32
pool1_width = int(map_width / 2)
pool1_height = int(map_height / 2)

# Layer 2
conv2_out_num = 64
pool2_width = int(pool1_width / 2)
pool2_height = int(pool1_height / 2)

# Layer 3
conv3_out_num = 64
pool3_width = int(pool2_width / 2)
pool3_height = int(pool2_height / 2)

# Output
num_acts = 5

# Reinforcement learning parameters
gamma = 0.9
batch_size = 10
epoch = 1000
l_rate = 0.01


#––––––––––––––––––––––––––––Input Data–––––––––––––––––––––––––––––––––#
def get_data():
    # Load the data from csv file
    if os.path.isfile(pickle_file):
        data = pickle.load(open(pickle_file, "rb"))
        print('pickle loaded')
    else:
        print('no pickle file')

    # Convert from (s, a, r) format to (s, a, r, s_)
    # TODO: Is this discarding important information? (the last turn is discarded?)
    data_s = []
    for i, (s, a, r) in enumerate(data):
        if i + 1 < len(data):
            data_s.append((s, a, r, data[i+1][0]))      #data[i+1][0] is s_

    # Create minibatches
    minibatches = []
    s_batch = []
    a_batch = []
    r_batch = []
    s_batch_ = []

    random.shuffle(data_s)

    for i in range(0, int(len(data) / batch_size), batch_size):
        # TODO: check this section again
        # minibatches = [data_s.pop(random.randrange(len(data_s))) for _ in range(batch_size)]
        minibatches.append(data_s[i:i + batch_size])

    for minibatch in minibatches:
        # Separately store minibatches
        s_batch.append(np.array([single_data[0] for single_data in minibatch]))
        a_batch.append(np.array([single_data[1] for single_data in minibatch]))
        r_batch.append(np.array([single_data[2] for single_data in minibatch]))
        s_batch_.append(np.array([single_data[3] for single_data in minibatch]))

    return (s_batch, a_batch, r_batch, s_batch_)






print('Minibatches ready')

#––––––––––––––––––––––––––––---load weight–––––––––––––––––––––––––––––#
# Load weight
# TODO: Does this have to be after creating the network?


if os.path.isfile(weights_file):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "weights/model.ckpt")
    print('Weights loaded')
else:
  # Initialisation
  init = tf.global_variables_initializer()

  # Session
  sess = tf.Session()
  sess.run(init)





def create_network():
    # Placeholders for s
    s = tf.placeholder(tf.float32, shape=[None, map_width, map_height, num_chan])

    def conv_layer(in_data, in_chan, out_chan):
        w_conv = tf.Variable(tf.truncated_normal([3, 3, in_chan, out_chan], stddev=0.1))
        b_conv = tf.Variable(tf.constant(0.1, shape=[out_chan]))
        return tf.nn.relu(tf.nn.conv2d(in_data, w_conv, strides=[1,1,1,1], padding='SAME') + b_conv)

    def pooling_layer(conv):
        h_pool = tf.nn.max_pool(conv,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME')
        return h_pool

    def full_layer(in_data, num_neuron, num_out):
        w_full = tf.Variable(tf.truncated_normal([num_neuron, num_out]))
        b_full = tf.Variable(tf.constant(0.1, shape=[num_out]))
        return tf.matmul(in_data, w_full) + b_full

    # Convolutional layer 1, out:map_width x map_height x conv1_num_chan
    conv1 = conv_layer(s, num_chan, conv1_out_num)

    # Pooling layer 1, out: pool1_width x pool1_height x conv1_out_num
    pool1 = pooling_layer(conv1)

    # Convolutional layer 2, out:pool1_width x pool1_height x conv2_out_num
    conv2 = conv_layer(pool1, conv1_out_num, conv2_out_num)

    # Pooling layer 2, out: pool2_width x pool2_height x conv2_out_num
    pool2 = pooling_layer(conv2)

    # Convolutional layer 3, out:pool2_width x pool2_height x conv3_out_num
    conv3 = conv_layer(pool2, conv2_out_num, conv3_out_num)

    # Pooling layer 3, out: pool3_width x pool3_height x conv3_out_num
    pool3 = pooling_layer(conv3)

    # Fully connected layer
    q_s = full_layer(tf.layers.flatten(pool3), pool3_width * pool3_height * conv3_out_num, num_acts)

    return q_s, s


def train_network(q_s, s, sess, batch):
    # Placeholders
    a = tf.placeholder(tf.float32, shape=[None, num_acts])
    y = tf.placeholder(tf.float32, shape=[None])
    r = tf.placeholder(tf.float32, shape=[None])
    q_s_a = tf.reduce_sum(tf.multiply(q_s, a))
    loss = tf.reduce_mean(tf.square(y - q_s_a))
    train_step = tf.train.AdamOptimizer(l_rate).minimize(loss)

    (s_batch, a_batch, r_batch, s_batch_) = batch
    y_batch = []

    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        print('i: %d' % i)
        s_batch_ = np.array(s_batch_)
        s_batch_ = np.reshape(s_batch_, (batch_size, map_width, map_height, num_chan))

        q_s_a_t = q_s.eval(feed_dict={s: s_batch_})

        # TODO: something wrong the way r_batch is created
        for i in range(batch_size):
            y_batch.append(r_batch[0][i] + gamma * np.max(q_s_a_t))

        # TODO: reshape inside the get_data()
        a_batch = np.array(a_batch)
        a_batch = np.reshape(a_batch, (batch_size, num_acts))
        train_step.run(feed_dict={
            y: y_batch,
            a: a_batch,
            s: s_batch_})

batches = get_data()
sess = tf.InteractiveSession()
q_s, s = create_network()
train_network(q_s, s, sess, batches)

# Save the weights
saver = tf.train.Saver()
saver.save(sess, weights_file)
print('Weights saved')

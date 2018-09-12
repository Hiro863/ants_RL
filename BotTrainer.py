'''
CNN for training the bot

Aim: Learns Q(s, a)

Input:
map_width x map_height x num_chan

Channels:   - my ants
            - enemy ants
            - food
            - water
            - my hills
            - enemy hills
            - current ant

Layers:
3 CNNs and pooling layers and 1 fully connected layer

Output: Q(s, a), approximating Q*


DQN:
s: state, numpy array, [input data map_width x map_height x num_chan]
a: action, numpy array, [num_acts]
r: reward, float
q_s: Q(s, a), numpy array [num_acts] (for all a')
q_s_a: Q (s, a), float
q_opt: Q*(s, a), float
q_: Q*(s', a'), numpy array [num_acts] (for all a)


'''


import tensorflow as tf
import numpy as np
import random
import os.path

#–––––––––––––––––––––––––––––Hyperparameters––––––––––––––––––––––––––#
# File names
csv_file = 'data_set.csv'
weights_file = 'weights/model.ckpt'

# Input
map_width = 100
map_height = 100
num_chan = 7

# Layer 1
# TODO: Map size might need to be adjusted
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
gamma = 0.9
batch_size = 10
epoch = 1000
l_rate = 0.01


#––––––––––––––––––––––––––––Input Data–––––––––––––––––––––––––––––––––#

# Load the data from csv file
if os.path.isfile(csv_file):
    data = np.genfromtxt(csv_file, delimiter=',')
    print('csv loaded')
else:
    print('no csv file')

# Convert from (s, a, r) format to (s, a, r, s_)
# TODO: Is this discarding important information? (the last turn is discarded?)
data_s = []
for i, (s, a, r) in enumerate(data):
    if i + 1 < len(data):
        data_s.append((s, a, r, data[i+1]))

# Create minibatches
# TODO: is this compatible with unstable len(data)?
minibatch = random.sample(data_s, batch_size)

# Separately store minibatches
s_batch = np.array([single_data[0] for single_data in minibatch])
a_batch = np.array([single_data[1] for single_data in minibatch])
r_batch = np.array([single_data[2] for single_data in minibatch])
s_batch_ = np.array([single_data[3] for single_data in minibatch])

print('Minibatches ready')

#––––––––––––––––––––––––––––---load weight–––––––––––––––––––––––––––––#
# Load weight
saver = tf.train.Saver()

if os.path.isfile(weights_file):
    with tf.Session() as sess:
        saver.restore(sess, "weights/model.ckpt")
    print('Weights loaded')
else:
    # Initialisation
    init = tf.global_variables_initializer()

    # Session
    sess = tf.Session()
    sess.run(init)

#–––––––––––––––––––––––––––––––––––CNN–––––––––––––––––––––––––––––––––#

# Placeholders for inputs
s = tf.placeholder(tf.float32, shape=[None, map_width, map_height, num_chan])
s_ = tf.placeholder(tf.float32, shape=[None, map_width, map_height, num_chan])
a = tf.placeholder(tf.float32, shape=[None, num_acts])
r = tf.placeholder(tf.float32, shape=[None])


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
    h_conv3 = tf.nn.relu(tf.conv2d(h_pool2, w_conv3) + b_conv1)

# Pooling layer 3, out: pool3_width x pool3_height x conv3_out_num
with tf.name_scope('pool3') as scope:
    h_pool3 = tf.nn.max_pool(h_conv3,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')

# Fully connected layer
with tf.name_scope('full1') as scope:
    w_full = tf.truncated_normal([pool3_width * pool3_height * conv3_out_num, num_acts])
    b_full = tf.constant(0.1, [num_acts])
    q_s = tf.matmul(h_pool3, w_full) + b_full

# Loss function
q_s_a = tf.reduce_sum(tf.mul(q_s, a))
q_ = q_s.eval(feed_dict={s: s_})
loss = tf.reduce_mean(tf.square(r + gamma * np.max(q_) - q_s_a))

# Train step
train_step = tf.train.GradientDescentOptimizer(l_rate).minimize(loss)

# Initialisation
init = tf.global_variables_initializer()

#–––––––––––––––––––––––––––––––Training–––––––––––––––––––––––––––––––––#

# Training session
print('Starting a training session')
for i in range(epoch):
    sess.run(train_step, feed_dict={s: s_batch, s_: s_batch_, a: a_batch, r: r_batch})

    print('Epoch no. %d' % i)

# Save the weights
saver = tf.train.Saver()
saver.save(sess, "weights/model.ckpt")
print('Weights saved')

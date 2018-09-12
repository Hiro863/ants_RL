'''
Input: state s
Output: action a, as a one-hot vector
'''


import tensorflow as tf
import numpy as np
import random
from BotTrainer import create_network

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

        q_s, s = create_network()

        q = q_s.eval(feed_dict={s: s_in})
        a_index = np.argmax(q)

    a[a_index] = 1
    return a


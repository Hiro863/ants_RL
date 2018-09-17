'''
Input: state s
Output: action a, as a one-hot vector
'''

import tensorflow as tf
import numpy as np
import random
import os.path
from BotTrainer import create_network

# File names
weights_file = 'weights/model.ckpt'

# Input
input_size = 16
num_chan = 6

# Output
num_acts = 5

# Reinforcement learning parameters
init_epsilon = 0.3
fin_epsilon = 0.05
explore = 500


class DecisionMaker:
    def __init__(self):
        self.q_s, self.s, variables = create_network()
        self.is_weights = False
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Load weight
        if os.path.exists(weights_file):
            w_conv1, w_conv2, w_conv3, b_conv1, b_conv2, b_conv3, w_full, b_full = variables

            saver = tf.train.Saver({'w_conv1': w_conv1,
                                    'w_conv2': w_conv2,
                                    'w_conv3': w_conv3,
                                    'b-conv1': b_conv1,
                                    'b_conv2': b_conv2,
                                    'b_conv3': b_conv3,
                                    'w_full': w_full,
                                    'b_full': b_full})

            saver.restore(self.sess, weights_file)
            self.is_weights = True


    def make_decision(self, s_in):
        # action one-hot vector
        a = np.zeros(num_acts)
        s_in = np.reshape(s_in, (1, input_size, input_size, num_chan))

        # Move randomly if first time, else move according to learnt strategy
        if not self.is_weights:
            a_index = random.randrange(num_acts)
            a[a_index] = 1
            return a

        else:
            # epsilon to make sure further learning
            epsilon = init_epsilon

            # decide whether to explore or stick to the best known strategy
            if random.random() <= epsilon:
                a_index = random.randrange(num_acts)
            else:
                # TODO: Is this right?
                # scale down epsilon
                if epsilon > fin_epsilon:
                    epsilon -= (init_epsilon - fin_epsilon) / explore

                with self.sess:
                    q = self.q_s.eval(feed_dict={self.s: s_in})
                a_index = np.argmax(q)

            a[a_index] = 1
            return a
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
# TODO: get paths sorted
weights_file = '/Users/hiro/Documents/IT_and_Cognition/Scientific_Programming/aichallenge-epsilon/ants/ants_RL/tools/weights/model.ckpt'

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
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Load weight
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



    def make_decision(self, s_in):
        # action one-hot vector
        a = np.zeros(num_acts)



        # Reshape input
        s_in = np.reshape(s_in, (1, input_size, input_size, num_chan))

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

            q = self.sess.run(self.q_s, feed_dict={self.s: s_in})
            a_index = np.argmax(q)

        a[a_index] = 1

        return a
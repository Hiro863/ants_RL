'''
Input: state s
Output: action a, as a one-hot vector
'''

import tensorflow as tf
import numpy as np
import random
import os.path
from BotTrainer import create_network
from pickle import Pickler, Unpickler

# File names
# TODO: get paths sorted
weights_file = os.path.join(os.path.dirname(__file__), 'tools/weights/model.ckpt')
pickle_file = 'epsilon.p'


# Input
input_size = 16
num_chan = 6

# Output
num_acts = 5

# Reinforcement learning parameters
init_epsilon = 1.0
fin_epsilon = 0.05
explore = 500000


class DecisionMaker:
    def __init__(self):
        self.q_s, self.s, self.keep_prob = create_network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.is_weights = False

        # Load weight
        if os.path.exists(weights_file):
            saver = tf.train.Saver({'w_conv1': w_conv1,
                                    'w_conv2': w_conv2,
                                    'w_conv3': w_conv3,
                                    'b_conv1': b_conv1,
                                    'b_conv2': b_conv2,
                                    'b_conv3': b_conv3,
                                    'w_full': w_full,
                                    'b_full': b_full})

            saver.restore(self.sess, weights_file)
            self.is_weights = True

        # set epsilon
        self.epsilon = init_epsilon
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as f:
                unpickler = Unpickler(f)
                try:
                    self.epsilon = unpickler.load()
                except EOFError:
                    pass
        else:
            self.epsilon = init_epsilon

    def make_decision(self, s_in):
        # action one-hot vector
        a = np.zeros(num_acts)

        if self.is_weights:
            # Reshape input
            s_in = np.reshape(s_in, (1, input_size, input_size, num_chan))

            # decide whether to explore or stick to the best known strategy
            if random.random() <= self.epsilon:
                a_index = random.randrange(num_acts)
            else:
                q = self.sess.run(self.q_s, feed_dict={self.s: s_in, self.keep_prob: 1.0})
                a_index = np.argmax(q)

        else:
            a_index = random.randint(0, num_acts - 1)

        a[a_index] = 1

        # scale down epsilon
        if self.epsilon > fin_epsilon:
            self.epsilon -= (init_epsilon - fin_epsilon) / explore
            self.save_epsilon()
        return a

    def save_epsilon(self):
        with open(pickle_file, 'wb') as f:
            Pickler(f).dump(self.epsilon)

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
weights_file = os.path.join(os.path.dirname(__file__), 'weights/model.ckpt')
pickle_dir = 'pickle_files'
pickle_file_session_results = 'session_results.p'


# Input
input_size = 16
num_chan = 6

# Output
num_acts = 5

# Reinforcement learning parameters
init_epsilon = 1.0
fin_epsilon = 0.05
explore = 1000000


class DecisionMaker:
    def __init__(self):
        self.q_s, self.s, self.keep_prob = create_network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.is_weights = False
        self.count = 0

        # Load weight
        if os.path.exists(weights_file):
            saver = tf.train.Saver()
            saver.restore(self.sess, weights_file)
            self.is_weights = True

        # set epsilon
        pickle_path = os.path.join(pickle_dir, pickle_file_session_results)
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                unpickler = Unpickler(f)
                try:
                    self.epsilon, self.count = unpickler.load()
                except EOFError:
                    pass
        else:
            self.epsilon = init_epsilon

    def make_decision(self, s_in):
        # number of time it was called
        self.count += 1

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
            self.save_results()
        return a

    def save_results(self):
        pickle_path = os.path.join(pickle_dir, pickle_file_session_results)
        with open(pickle_path, 'wb') as f:
            Pickler(f).dump((self.epsilon, self.count))

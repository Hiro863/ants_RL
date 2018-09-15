'''
Input: state s
Output: action a, as a one-hot vector
'''

import numpy as np
import random
import os.path
from BotTrainer import create_network

#–––––––––––––––––––––––––––––Hyperparameters––––––––––––––––––––––––––#
# File names
csv_file = 'data_set.csv'
weights_file = 'weights/model.ckpt'

# Output
num_acts = 5

# Reinforcement learning parameters
init_epsilon = 1.0
fin_epsilon = 0.05
explore = 500


# TODO maybe create a class to avoid having to create network each time
class DecisionMaker():
    def __init__(self):
        # TODO:Load weights
        self.q_s, self.s = create_network()

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

    def make_decision(self, s_in):
        # action one-hot vector
        a = np.zeros(num_acts)

        # Move randomly if first time, else move according to learnt strategy
        if not os.path.isfile(weights_file):
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

                q = self.q_s.eval(feed_dict={self.s: s_in})
                a_index = np.argmax(q)

            a[a_index] = 1
            return a

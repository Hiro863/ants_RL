'''
Input: state s
Output: action a, as a one-hot vector
'''

import numpy as np
import random
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


def decision_maker(s_in):

    # epsilon to make sure further learning
    epsilon = init_epsilon

    # action one-hot vector
    a = np.zeros(num_acts)

    # decide whether to explore or stick to the best known strategy
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


from __future__ import division
from scipy import sparse
from pickle import Pickler, Unpickler
import numpy as np
import math
import os


class TrainingStorage:
    def __init__(self, file='training.p', remove=False):
        self.path = os.path.join(os.path.dirname(__file__), file)
        if remove and os.path.isfile(self.path):
            os.remove(self.path)

    def map_state(self, ants):
        # some data can be retrieved from map property
        dims = int(math.ceil(ants.rows / 8) * 8), int(math.ceil(ants.cols / 8) * 8)
        m = np.array(ants.map, dtype=np.dtype('b'))

        water = sparse.lil_matrix(dims, dtype=np.dtype('b'))
        for row, col in zip(*np.where(m == -4)):
            water[row, col] = 1

        food = sparse.lil_matrix(dims, dtype=np.dtype('b'))
        for row, col in zip(*np.where(m == -3)):
            food[row, col] = 1

        enemy_ants = sparse.lil_matrix(dims, dtype=np.dtype('b'))
        enemy_hills = sparse.lil_matrix(dims, dtype=np.dtype('b'))

        my_ants = sparse.lil_matrix(dims, dtype=np.dtype('b'))
        my_hills = sparse.lil_matrix(dims, dtype=np.dtype('b'))

        # rest is retrieved by iterating over ant_list
        for (row, col), owner in ants.ant_list.items():
            if owner != 0:
                enemy_ants[row, col] = 1
            else:
                my_ants[row, col] = 1

        # and over hill list
        for (row, col), owner in ants.hill_list.items():
            if owner != 0:
                enemy_hills[row, col] = 1
            else:
                my_hills[row, col] = 1

        return [my_ants, my_hills, enemy_ants, enemy_hills, water, food]

    def current_ant(self, dimensions, ant_loc):
        arow, acol = ant_loc
        current_ant = sparse.lil_matrix(dimensions, dtype=np.dtype('b'))
        current_ant[arow, acol] = 1

        return current_ant

    def state(self, ants, ant_loc):
        dims = int(math.ceil(ants.rows / 8) * 8), int(math.ceil(ants.cols / 8) * 8)
        current_ant = self.current_ant(dims, ant_loc)
        state = self.map_state(ants)
        state.insert(0, current_ant)

        return state

    def remember(self, state, action, reward, turn):
        with open(self.path, 'ab+') as f:
            data = state, action, reward, turn
            Pickler(f).dump(data)

    def items(self):
        with open(self.path, 'rb') as f:
            unpickler = Unpickler(f)
            try:
                while True:
                    sparse_state, action, reward, turn = unpickler.load()
                    state = []
                    for sparse_state_channel in sparse_state:
                        state.append(sparse_state_channel.toarray())

                    yield state, action, reward, turn
            except EOFError:
                pass

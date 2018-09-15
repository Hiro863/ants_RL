from scipy import sparse
from pickle import Pickler, Unpickler
import numpy as np
import math
import os


class TrainingData:
    def __init__(self, filename='training.p'):
        self.path = os.path.join(os.path.dirname(__file__), filename)

    def state(self, ants):
        # some data can be retrieved from map property
        m = np.array(ants.map, dtype=np.dtype('b'))
        water = sparse.lil_matrix(np.where(m == -4, 1, 0), dtype=np.dtype('b'))
        food = sparse.lil_matrix(np.where(m == -3, 1, 0), dtype=np.dtype('b'))

        dimensions = math.ceil(ants.rows / 8) * 8, math.ceil(ants.cols / 8) * 8
        enemy_ants = sparse.lil_matrix(dimensions, dtype=np.dtype('b'))
        enemy_hills = sparse.lil_matrix(dimensions, dtype=np.dtype('b'))

        my_ants = sparse.lil_matrix(dimensions, dtype=np.dtype('b'))
        my_hills = sparse.lil_matrix(dimensions, dtype=np.dtype('b'))

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

        return (my_ants, my_hills, enemy_ants, enemy_hills, water, food)

    def append(self, ants):
        with open(self.path, 'ab+') as f:
            state = self.map_state(ants)
            Pickler(f).dump(state)

    def items(self):
        with open(self.path, 'rb') as f:
            unpickler = Unpickler(f)
            try:
                while True:
                    state = []
                    for sparsemat in unpickler.load():
                        state.append(sparsemat.toarray())
                    yield state
            except EOFError:
                pass

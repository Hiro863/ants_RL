from __future__ import division
from scipy import sparse
from pickle import Pickler, Unpickler
import numpy as np
import os

input_size = 16

class TrainingStorage:
    def __init__(self, file='training.p', remove=False):
        self.path = os.path.join(os.path.dirname(__file__), file)
        if remove and os.path.isfile(self.path):
            os.remove(self.path)

    def state(self, ants, ant_loc):
        # some data can be retrieved from map property
        dims = (input_size, input_size)
        m = np.array(ants.map, dtype=np.dtype('b'))

        # convert the map to input_size x input_size window
        new_m = self.map_convert(ant_loc, ants, m)

        water = sparse.lil_matrix(dims, dtype=np.dtype('b'))
        for row, col in zip(*np.where(new_m == -4)):
            water[row, col] = 1

        food = sparse.lil_matrix(dims, dtype=np.dtype('b'))
        for row, col in zip(*np.where(new_m == -3)):
            food[row, col] = 1

        enemy_ants = sparse.lil_matrix(dims, dtype=np.dtype('b'))
        enemy_hills = sparse.lil_matrix(dims, dtype=np.dtype('b'))

        my_ants = sparse.lil_matrix(dims, dtype=np.dtype('b'))
        my_hills = sparse.lil_matrix(dims, dtype=np.dtype('b'))

        # rest is retrieved by iterating over ant_list
        for (row, col), owner in ants.ant_list.items():
            in_range, new_row, new_col = self.loc_convert(ant_loc, ants, (row, col))
            if in_range:
                if owner != 0:
                    enemy_ants[new_row, new_col] = 1
                else:
                    my_ants[new_row, new_col] = 1

        # and over hill list
        for (row, col), owner in ants.hill_list.items():
            in_range, new_row, new_col = self.loc_convert(ant_loc, ants, (row, col))
            if in_range:
                if owner != 0:
                    enemy_hills[new_row, new_col] = 1
                else:
                    my_hills[new_row, new_col] = 1

        return [my_ants, my_hills, enemy_ants, enemy_hills, water, food]

    def current_ant(self, dimensions, ant_loc):
        arow, acol = ant_loc
        current_ant = sparse.lil_matrix(dimensions, dtype=np.dtype('b'))
        current_ant[arow, acol] = 1

        return current_ant

    def remember(self, state, action, reward, label, turn):
        with open(self.path, 'ab+') as f:
            data = state, action, reward, label, turn
            Pickler(f).dump(data)

    def items(self):
        with open(self.path, 'rb') as f:
            unpickler = Unpickler(f)
            try:
                while True:
                    sparse_state, action, reward, label, turn = unpickler.load()
                    state = []
                    for sparse_state_channel in sparse_state:
                        state.append(sparse_state_channel.toarray())

                    yield state, action, reward, label, turn
            except EOFError:
                pass


    def map_convert(self, ant_loc, ants, m):
        # determine which squares are visible to the ant
        # precalculate squares around an ant to set as visible
        # ant is located at (8, 8)
        # TODO: I ignored the corners (so there is no -1)

        a_row, a_col = ant_loc
        new_map = np.zeros(shape=(input_size, input_size), dtype=np.dtype('b'))

        # find origin
        if a_row - 8 >= 0:  # inside the map
            o_row = a_row - 8
        else:  # outside the map
            o_row = ants.rows + (a_row - 8)
        if a_col - 8 >= 0:  # inside the map
            o_col = a_col - 8
        else:  # outside the map
            o_col = ants.cols + (a_col - 8)

        for row in range(input_size):
            for col in range(input_size):

                m_row = o_row + row
                m_col = o_col + col

                if m_row >= ants.rows:
                    m_row = m_row - ants.rows
                if m_col >= ants.cols:
                    m_col = m_col - ants.cols

                new_map[row, col] = m[m_row, m_col]  # copy the content

        return new_map


    def loc_convert(self, ant_loc, ants, m_loc):
        a_row, a_col = ant_loc
        m_row, m_col = m_loc

        # find origin
        if a_row - 8 >= 0:                      # inside the map
            o_row = a_row - 8
        else:                                   # outside the map
            o_row = ants.rows + (a_row - 8)
        if a_col - 8 >= 0:                      # inside the map
            o_col = a_col - 8
        else:                                   # outside the map
            o_col = ants.cols + (a_row - 8)

        row = m_row - o_row
        col = m_col - o_col
        if row > 15 or row < 0 or col > 15 or col < 0:      # outside the roi
            return False, None, None
        else:
            return True, row, col


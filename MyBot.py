#!/usr/bin/env python
from __future__ import division
from ants import Ants
import os
from storage import TrainingStorage
from antutils import logexcept, DEBUG_LOG
from BotDecisionMaker import DecisionMaker
import numpy as np
from antutils import log


class MyBot:
    def __init__(self, storage):
        self.storage = storage
        self.dmake = DecisionMaker()
        self.directions = {
            0: 'n',
            1: 'e',
            2: 's',
            3: 'w',
            4: 'r'
        }
        self.history = {}
        self.history_length = 3
        self.turn = 0

    def do_setup(self, ants):
        pass

    def think(self, sparse_state):
        state_list = []
        for sparse_state_channel in sparse_state:
            state_channel = sparse_state_channel.toarray()
            state_list.append(state_channel)
        state = np.stack(state_list)
        decision = self.dmake.make_decision(state)

        return decision

    def reward(self, state):
        current_ant, my_ants, my_hills, enemy_ants, enemy_hills, water, food = state
        reward = (my_ants == 1).sum()

        return reward

    def append_history(self, state, action):
        if self.turn in self.history:
            self.history[self.turn].append((state, action))
        else:
            self.history[self.turn] = [(state, action)]

        expired = (key for key in self.history.keys() if key <= (self.turn - self.history_length))
        for turn in expired:
            del self.history[turn]

    @logexcept
    def do_turn(self, ants):
        self.turn += 1
        for ant_loc in ants.my_ants():
            state = self.storage.state(ants, ant_loc)
            direction_onehot = self.think(state)
            direction = self.directions[np.where(direction_onehot == 1)[0][0]]

            # remember what have we done this turn
            self.append_history(state, direction_onehot)

            if direction != 'r':
                ants.issue_order((ant_loc, direction))

            if ants.time_remaining() < 10:
                break

        # we need to know the outcome before we calculate the reward
        # thats why only previous turn is stored
        offset = 2
        if len(self.history) > offset:
            for prev_state, prev_action in self.history[self.turn - offset]:
                self.storage.remember(
                    prev_state, prev_action,
                    self.reward(state), self.turn - offset
                )


if __name__ == '__main__':
    @logexcept
    def start():
        try:
            os.remove(DEBUG_LOG)
        except OSError:
            pass
        storage = TrainingStorage(remove=True)
        Ants.run(MyBot(storage))

    try:
        import psyco
        psyco.full()
    except ImportError:
        pass

    try:
        start()
    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')

#!/usr/bin/env python
from __future__ import division
from ants import Ants
import os
from storage import TrainingStorage
from antutils import logexcept, DEBUG_LOG, log
from BotDecisionMaker import DecisionMaker
import numpy as np
from tracking import Tracking

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
        self.history_length = 2
        self.turn = 0
        self.tracking = None

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

    def reward(self, food):
        # if both food_found and killed is True then reward = -100
        #if is_killed:
        #    reward = -100
        reward = food * 100

        return reward

    def append_history(self, state, action, label, future_food):
        if self.turn in self.history:
            self.history[self.turn].append((state, action, label, future_food))
        else:
            self.history[self.turn] = [(state, action, label, future_food)]

        expired = (key for key in self.history.keys() if key <= (self.turn - self.history_length))
        for turn in list(expired):
            del self.history[turn]

    @logexcept
    def do_turn(self, ants):
        self.turn += 1
        if not self.tracking:
            self.tracking = Tracking()
        self.tracking.update(ants)

        for ant_loc in ants.my_ants():
            state = self.storage.state(ants, ant_loc)
            direction_onehot = self.think(state)
            direction = self.directions[np.where(direction_onehot == 1)[0][0]]


            if direction != 'r':
                new_loc = ants.destination(ant_loc, direction)
            else:
                new_loc = ant_loc
            log((self.turn, 'Moving ant ', ant_loc, ' to ', new_loc))

            # remember what have we done this turn
            label = self.tracking.loc_to_ants[ant_loc]
            future_food = self.tracking.adjacent_food(new_loc, ants)
            self.append_history(state, direction_onehot, label, future_food)

            if direction != 'r':
                self.tracking.move_ant(ant_loc, direction, ants)

            # TODO: how often are we running out of time?
            if ants.time_remaining() < 10:
                log(('timeout'))
                break


        # we need to know the outcome before we calculate the reward
        # thats why only previous turn is stored
        offset = 1
        if len(self.history) > offset:
            for prev_state, prev_action, prev_label, food in self.history[self.turn - offset]:
                self.storage.remember(
                    prev_state, prev_action,
                    self.reward(food), prev_label,
                    self.turn - offset
                )

        self.dmake.save_epsilon()


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

#!/usr/bin/env python
from __future__ import division
from ants import Ants
import os
from training import TrainingData
from antutils import logexcept, DEBUG_LOG, log
from BotDecisionMaker import DecisionMaker
import numpy as np


class MyBot:
    def __init__(self, trainingdata):
        self.traindat = trainingdata
        self.dmake = DecisionMaker()
        self.directions = {
            0: 'n',
            1: 'e',
            2: 's',
            3: 'w',
            4: 'r'
        }

    def do_setup(self, ants):
        pass

    @logexcept
    def do_turn(self, ants):
        for ant_loc in ants.my_ants():
            state = self.traindat.state(ants, ant_loc)
            dense = []
            for sparse in state:
                d = sparse.toarray()
                dense.append(d)
            exp = np.stack(dense)
            decision = self.dmake.make_decision(exp)
            dir = self.directions[np.where(decision == 1)[0][0]]
            if dir != 'r':
                new_loc = ants.destination(ant_loc, dir)
                (r1, c1), (r2, c2) = ant_loc, new_loc
                if not ants.passable(new_loc):
                    log("Invalid order (%d, %d => %d, %d)." % (r1, c1, r2, c2))
                else:
                    log("Order issued (%d, %d => %d, %d)." % (r1, c1, r2, c2))

                ants.issue_order((ant_loc, dir))

            self.traindat.append(state=state)
            if ants.time_remaining() < 10:
                break


if __name__ == '__main__':
    @logexcept
    def start():
        try:
            os.remove(DEBUG_LOG)
        except OSError:
            pass
        traindat = TrainingData(remove=True)
        Ants.run(MyBot(traindat))

    try:
        import psyco
        psyco.full()
    except ImportError:
        pass

    try:
        start()
    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')

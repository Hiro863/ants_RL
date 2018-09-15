#!/usr/bin/env python
from ants import Ants
from pprint import pprint
import traceback
import os
import random
import pickle

DEBUG_LOG = os.path.join(os.path.dirname(__file__), "debug.txt")
i = 0


def logexcept(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            with open(DEBUG_LOG, 'a+') as f:
                f.write(str(e))
                f.write(traceback.format_exc())
    return wrapper


def log(data):
    with open(DEBUG_LOG, 'a+') as f:
        pprint(data, f)


@logexcept
def save_ants_obj(ants):
    global i
    filename = os.path.join(os.path.dirname(__file__), "/ants_obj/ants_" + str(i) + ".p")
    pickle.dump(ants, open(filename), "wb")
    i += 1


class MyBot:
    def __init__(self):
        pass

    def do_setup(self, ants):
        pass

    @logexcept
    def do_turn(self, ants):
        for ant_loc in ants.my_ants():
            directions = ('n', 'e', 's', 'w')
            found_passable = False
            while not found_passable:
                direction = random.choice(directions)
                new_loc = ants.destination(ant_loc, direction)
                found_passable = ants.passable(new_loc)

                if found_passable:
                    ants.issue_order((ant_loc, direction))

            if ants.time_remaining() < 10:
                break


if __name__ == '__main__':
    try:
        import psyco
        psyco.full()
    except ImportError:
        pass

    try:
        try:
            os.remove(DEBUG_LOG)
        except OSError:
            pass
        Ants.run(MyBot())
    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')

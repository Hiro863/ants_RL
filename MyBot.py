#!/usr/bin/env python
from ants import Ants
from pprint import pprint
import traceback
import os

DEBUG_LOG = os.path.join(os.path.dirname(__file__), "debug.txt")


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


class MyBot:
    def __init__(self):
        pass

    def do_setup(self, ants):
        pass

    @logexcept
    def do_turn(self, ants):
        for ant_loc in ants.my_ants():
            log(ants.my_ants()[0])
            directions = ('n', 'e', 's', 'w')
            for direction in directions:
                new_loc = ants.destination(ant_loc, direction)
                if (ants.passable(new_loc)):
                    ants.issue_order((ant_loc, direction))
                    break
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

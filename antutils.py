import os
import traceback
from pprint import pprint
import pickle

i = 0
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


@logexcept
def save_ants_obj(ants):
    global i
    filename = os.path.join(os.path.dirname(__file__), "/ants_obj/ants_" + str(i) + ".p")
    pickle.dump(ants, open(filename), "wb")
    i += 1

from antutils import log

'''
Keeps track of which ant moved where
Each ant is identified by integer starting from 0
Needs to be updated at the beginning of every turn

'''
from math import sqrt

class Tracking:
    def __init__(self):
        self.loc_to_ants = {}
        self.ants_to_loc = {}
        self.num_ants = 0
        self.last_turn_moves = []

    def move_ant(self, loc, direc, ants):
        # store the moves that will be applied later
        new_loc = ants.destination(loc, direc)

        # procceed only if ant wants to move into accessible location (anything but not water)
        if (ants.passable(new_loc)):
            self.last_turn_moves.append((loc, new_loc, direc))
            ants.issue_order((loc, direc))
            return self.found_food(new_loc, ants)

        return False

    def apply_last_moves(self):
        # apply the stored moves
        loc_to_ants_new = {}
        for old_loc, new_loc, direc in self.last_turn_moves:
            # Pop the ant label from dictionary and save it to new one
            ant = self.loc_to_ants.pop(old_loc)
            loc_to_ants_new[new_loc] = ant

            # Update ants_to_loc
            self.ants_to_loc[ant] = new_loc

        # Add ants that have not moved to the new dictionary
        loc_to_ants_new.update(self.loc_to_ants)
        # Make that dictionary a current one
        self.loc_to_ants = loc_to_ants_new
        # Forget moves for this turn
        self.last_turn_moves = []

    def update(self, ants):
        my_ants = ants.my_ants()
        self.apply_last_moves()
        # if the ant is no longer in my ants, delete it
        self.ants_to_loc = {ant: loc for ant, loc in self.ants_to_loc.items() if loc in my_ants}
        self.loc_to_ants = {loc: ant for loc, ant in self.loc_to_ants.items() if loc in my_ants}

        # if the ant is not yet in the dictionary, add it
        for ant_loc in my_ants:
            if ant_loc not in self.loc_to_ants:
                self.loc_to_ants[ant_loc] = self.num_ants
                self.ants_to_loc[self.num_ants] = ant_loc
                self.num_ants += 1

    def found_food(self, new_loc, ants):
        # check if it found the food
        # new_loc is destination of the

        # create a list of visible foods
        visible_foods = []
        for food in ants.food():
            if self.visible_to_ant(food):
                visible_foods.append(food)

        # check if the ants steps on foods
        food_area = []
        adjacent = ((-1, 0),
                    (0, 1),
                    (1, 0),
                    (0, -1))
        for f_r, f_c in visible_foods:
            for a_r, a_c in adjacent:
                food_area.append((f_r + a_r, f_c, a_c))

        if new_loc in food_area:
            return True
        else:
            return False

    def killed(self, label):

        return is_killed


    def visible_to_ant(self, loc, ant_loc, ants):
        # determine which squares are visible to the ant

        if ants.vision == None:
            if not hasattr(ants, 'vision_offsets_2'):
                # precalculate squares around an ant to set as visible
                ants.vision_offsets_2 = []
                mx = int(sqrt(ants.viewradius2))
                for d_row in range(-mx, mx + 1):
                    for d_col in range(-mx, mx + 1):
                        d = d_row ** 2 + d_col ** 2
                        if d <= ants.viewradius2:
                            ants.vision_offsets_2.append((
                                d_row % ants.rows - ants.rows,
                                d_col % ants.cols - ants.cols
                            ))

            # set all spaces as not visible
            # loop through ants and set all squares around ant as visible
            ants.vision = [[False] * ants.cols for row in range(ants.rows)]
            row, col = ant_loc
            for v_row, v_col in ants.vision_offsets_2:
                ants.vision[row + v_row][col + v_col] = True
        row, col = loc
        return ants.vision[row][col]

# Debug
'''''''''''
import random
class Ants:
    def __init__(self):
        self.my_ants_list = []


    def my_ants(self):
        return self.my_ants_list

    def issue_order(self, order):
        loc, direc = order
        if direc == 'n':
            new_loc = (loc[0], loc[1] + 1)
        elif direc == 'e':
            new_loc = (loc[0] + 1, loc[1])
        elif direc == 's':
            new_loc = (loc[0], loc[1] - 1)
        else:
            new_loc = (loc[0] - 1, loc[1])

        self.my_ants_list = [v for v in self.my_ants_list if not v == loc]
        self.my_ants_list.append(new_loc)



ants = Ants()
tracker = Tracking()

for turn in range(100):
    print('---------------------------------')
    print('turn: %d' % turn)
    # Create random ants
    created = []
    for _ in range(random.randint(0, 3)):
        new_loc = (random.randint(0,20), random.randint(0,20))
        if new_loc not in ants.my_ants_list:
            ants.my_ants_list.append(new_loc)
            created.append(new_loc)

    # Kill random ants
    killed = []
    for _ in range(random.randint(0,3)):
        if len(ants.my_ants_list)-1 > 0:
            index = random.randint(0,len(ants.my_ants_list)-1)
            killed.append(ants.my_ants_list[index])
            del ants.my_ants_list[index]

    for loc in created:
        if loc not in killed:
            r, c = loc
            print('Created: (%d, %d)' % (r, c))

    for loc in killed:
        if loc not in created:
            r, c = loc
            print('Killed: (%d, %d)' % (r, c))

    tracker.update(ants)
    print(ants.my_ants_list)
    print(tracker.loc_to_ants)
    print(tracker.ants_to_loc)

    if len(ants.my_ants_list) > 0:
        loc = ants.my_ants_list[len(ants.my_ants_list)-1]
    direc = 'n'
    r, c = loc
    tracker.move_ant(loc, direc, ants)
    print('(%d, %d) moved' % (r, c))
'''''''''''

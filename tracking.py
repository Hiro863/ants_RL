from antutils import log

'''
Keeps track of which ant moved where
Each ant is identified by integer starting from 0
Needs to be updated at the beginning of every turn

'''
from math import sqrt
FOOD = -3

class Tracking:
    def __init__(self):
        self.loc_to_ants = {}
        self.ants_to_loc = {}
        self.num_ants = 0
        self.last_turn_moves = []
        self.killed = []

    def move_ant(self, loc, direc, ants):
        # store the moves that will be applied later
        new_loc = ants.destination(loc, direc)

        # procceed only if ant wants to move into accessible location (anything but not water)
        if (ants.passable(new_loc)):
            self.last_turn_moves.append((loc, new_loc, direc))
            ants.issue_order((loc, direc))

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

        # check if killed
        for ant_loc in self.loc_to_ants:
            if ant_loc not in my_ants:
                self.killed.append(self.loc_to_ants[ant_loc])

        # if the ant is no longer in my ants, delete it
        self.ants_to_loc = {ant: loc for ant, loc in self.ants_to_loc.items() if loc in my_ants}
        self.loc_to_ants = {loc: ant for loc, ant in self.loc_to_ants.items() if loc in my_ants}

        # if the ant is not yet in the dictionary, add it
        for ant_loc in my_ants:
            if ant_loc not in self.loc_to_ants:
                self.loc_to_ants[ant_loc] = self.num_ants
                self.ants_to_loc[self.num_ants] = ant_loc
                self.num_ants += 1

    def iter_adjacent(self, loc):
        adjacents = ((-1, 0), (0, 1), (1, 0), (0, -1))

        for vector in adjacents:
            drow, dcol = vector
            row, col = loc

            yield row + drow, col + dcol

    def adjacent_food(self, new_loc, ants):
        # check if there is food around
        # new_loc is destination of the ant
        food = 0
        for r, c in self.iter_adjacent(new_loc):
            if ants.map[r % ants.rows][c % ants.cols] == FOOD:
                food += 1

        return food

    def is_killed(self, label):
        if label in self.killed:
            return 1
        else:
            return 0

    def visible_to_ant(self, loc, ant_loc, ants):
        # determine which squares are visible to the ant
        rows = ants.rows
        cols = ants.cols
        viewradius2 = ants.viewradius2

        # precalculate squares around an ant to set as visible
        vision_offsets_2 = []
        mx = int(sqrt(viewradius2))
        for d_row in range(-mx, mx + 1):
            for d_col in range(-mx, mx + 1):
                d = d_row ** 2 + d_col ** 2
                if d <= viewradius2:
                    vision_offsets_2.append((
                        d_row % rows - rows,
                        d_col % cols - cols
                    ))

        # set all spaces as not visible
        # loop through ants and set all squares around ant as visible
        vision = [[False] * cols for row in range(rows)]
        row, col = ant_loc
        for v_row, v_col in vision_offsets_2:
            vision[row + v_row][col + v_col] = True

        row, col = loc
        return vision[row][col]

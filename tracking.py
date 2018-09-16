'''
Keeps track of which ant moved where
Each ant is identified by integer starting from 0
Needs to be updated at the beginning of every turn

'''

class Tracking:
    def __init__(self):
        self.loc_to_ants = {}
        self.ants_to_loc = {}
        self.num_ants = 0

    def move_ant(self, loc, direc, ants):
        # Get new location of the moved ant
        if direc == 'n':
            new_loc = (loc[0], loc[1] + 1)
        elif direc == 'e':
            new_loc = (loc[0] + 1, loc[1])
        elif direc == 's':
            new_loc = (loc[0], loc[1] - 1)
        else:
            new_loc = (loc[0] - 1, loc[1])

        # Update loc_to_ants
        ant = self.loc_to_ants[loc]
        del self.loc_to_ants[loc]
        self.loc_to_ants[new_loc] = ant

        # Update ants_to_loc
        self.ants_to_loc[ant] = new_loc

        # Move the ant
        ants.issue_order((loc, direc))

    def update(self, ants):
        my_ants = ants.my_ants()
        # if the ant is no longer in my ants, delete it
        self.ants_to_loc = {ant: loc for ant, loc in self.ants_to_loc.items() if loc in my_ants}
        self.loc_to_ants = {loc: ant for loc, ant in self.loc_to_ants.items() if loc in my_ants}

        # if the ant is not yet in the dictionary, add it
        for ant_loc in my_ants:
            if ant_loc not in self.loc_to_ants:
                self.loc_to_ants[ant_loc] = self.num_ants
                self.ants_to_loc[self.num_ants] = ant_loc
                self.num_ants += 1



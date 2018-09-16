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
            new_loc = (loc[0] - 1, loc[1])
        elif direc == 'e':
            new_loc = (loc[0], loc[1] + 1)
        elif direc == 's':
            new_loc = (loc[0] + 1, loc[1])
        else:
            new_loc = (loc[0], loc[1] - 1)

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
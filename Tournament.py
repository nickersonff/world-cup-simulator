import numpy as np


class Tornment:

    def __init__(self, teams_list, ranks=None):
        self.teams_list = teams_list
        self.ranks = ranks

    def match_type_dist(self):
        match_type = {}
        match_type['low'] = 0
        match_type['reg'] = 0
        match_type['high'] = 0
        for mm in self.matches:
            rkk1 = self.ranks[mm.result.teams[0]]['rank']
            rkk2 = self.ranks[mm.result.teams[1]]['rank']
            #print("rankings: ", rkk1, rkk2)
            if (rkk1 < 8 and rkk2 < 8):
                match_type['high'] = match_type['high'] + 1
            elif (rkk1 >= 8 and rkk2 >= 8):
                match_type['low'] = match_type['low'] + 1
            else:
                match_type['reg'] = match_type['reg'] + 1

        return match_type

    def match_rank_index(self):
        match_rank_index = [0 for ii in self.matches]
        for mmi, mm in enumerate(self.matches):
            # colocar o rating ??
            rkk1 = 1.0 - \
                float(min([self.ranks[mm.result.teams[0]]['rank'], 50])-1)/50.0
            rkk2 = 1.0 - \
                float(min([self.ranks[mm.result.teams[1]]['rank'], 50])-1)/50.0
            match_rank_index[mmi] = np.sqrt(rkk1*rkk2)

        return match_rank_index

    def match_rank_distance(self):
        match_rank_distance = [0 for ii in self.matches]
        for mmi, mm in enumerate(self.matches):
            rkk = self.ranks[mm.result.teams[0]]['rank'] - \
                self.ranks[mm.result.teams[1]]['rank']
            if (rkk < 0):
                rkk = -1*rkk
            match_rank_distance[mmi] = rkk

        return match_rank_distance

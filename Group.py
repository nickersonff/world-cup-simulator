import numpy as np


class Group:

    matches = []
    teams = []
    position = []

    def __init__(self, teams, matches=None):
        if (matches == None):
            self.matches = []
        else:
            self.matches = matches
        self.teams = teams

    def draw(self, ngroup=4):
        if ngroup == 1:
            self.position = list(np.random.permutation(self.teams))
        else:
            pots = list()
            num = len(self.teams)//ngroup
            for gr in range(ngroup):
                pots.append(list(np.random.permutation(
                    self.teams[gr*num:(gr+1)*num])))
            self.position = []
            for i in range(len(pots[0])):
                aux = list()
                for j in range(len(pots)):
                    aux.append(pots[j][i])
                self.position.extend(
                    list(np.random.permutation(aux)))

    def compute(self):
        points = {}
        for tt in self.teams:
            points[tt] = [0, 0, 0]
        for mm in self.matches:
            tt = mm.result.teams[0]
            if (tt in self.teams):
                points[tt][0] += mm.result.points[0]
                points[tt][1] += mm.result.goals[0]-mm.result.goals[1]
                points[tt][2] += mm.result.goals[0]
            tt = mm.result.teams[1]
            if (tt in self.teams):
                points[tt][0] += mm.result.points[1]
                points[tt][1] += mm.result.goals[1]-mm.result.goals[0]
                points[tt][2] += mm.result.goals[1]

            points2 = np.argsort([points[tt][2] for tt in self.teams])[::-1]
            self.position = [self.teams[pp] for pp in points2]
            points2 = np.argsort([points[tt][1] for tt in self.position])[::-1]
            self.position = [self.position[pp] for pp in points2]
            points2 = np.argsort([points[tt][0] for tt in self.position])[::-1]
            self.position = [self.position[pp] for pp in points2]

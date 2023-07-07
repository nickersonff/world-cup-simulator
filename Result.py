import numpy as np
from Params import Params


class Result:

    winner = ''
    loser = ''

    teams = ['', '']
    ranks = [0, 0]

    goals = [-1, -1]
    points = [-1, -1]

    def __init__(self, teams=None, ranks=None):
        self.teams = teams
        self.ranks = ranks
        if (self.teams == None):
            self.teams = ['', '']
        if (self.ranks == None):
            self.ranks = [0, 0]

    def set_teamHome(self, team, rank=0):
        self.teams[0] = team
        self.ranks[0] = rank

    def set_teamAway(self, team, rank=0):
        self.teams[1] = team
        self.ranks[1] = rank

    def play(self, method='rankn', gain=None):

        self.goals = np.random.poisson(1.5, size=2)
        lamb0 = Params.ALFA * (float(self.ranks[0]['elo-rating'])
                               / (float(self.ranks[0]['elo-rating']) + float(self.ranks[1]['elo-rating'])))
        lamb1 = Params.ALFA * (self.ranks[1]['elo-rating'] / (self.ranks[0]
                                                              ['elo-rating'] + self.ranks[1]['elo-rating']))
        self.goals[0] = np.random.poisson(lamb0)
        self.goals[1] = np.random.poisson(lamb1)

        if (self.goals[0] > self.goals[1]):
            self.winner = self.teams[0]
            self.loser = self.teams[1]
            self.points = [3, 0]
        elif (self.goals[0] < self.goals[1]):
            self.winner = self.teams[1]
            self.loser = self.teams[0]
            self.points = [0, 3]
        else:
            qual = np.random.randint(2)
            self.winner = self.teams[qual]
            self.loser = self.teams[1-qual]
            self.points = [1, 1]

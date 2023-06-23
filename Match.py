from Result import Result


class Match:

    number = 0
    fromHome = ('n')  # winner,loser,position/draw,notdefined
    fromAway = ('n')
    result = []

    def __init__(self, number, fromHome, fromAway):

        self.number = number
        self.fromHome = fromHome
        self.fromAway = fromAway

    def setup(self, matches=None, groups=None, ranks=None):

        self.result = Result()

        if (ranks == None):

            if self.fromHome[0] == 'w':
                self.result.set_teamHome(
                    matches[self.fromHome[1]].result.winner)
            elif self.fromHome[0] == 'l':
                self.result.set_teamHome(
                    matches[self.fromHome[1]].result.loser)
            elif self.fromHome[0] == 'p':
                self.result.set_teamHome(
                    groups[self.fromHome[1]].position[self.fromHome[2]])

            if self.fromAway[0] == 'w':
                self.result.set_teamAway(
                    matches[self.fromAway[1]].result.winner)
            elif self.fromAway[0] == 'l':
                self.result.set_teamAway(
                    matches[self.fromAway[1]].result.loser)
            elif self.fromAway[0] == 'p':
                self.result.set_teamAway(
                    groups[self.fromAway[1]].position[self.fromAway[2]])

        else:

            if self.fromHome[0] == 'w':
                self.result.set_teamHome(
                    matches[self.fromHome[1]].result.winner, ranks[matches[self.fromHome[1]].result.winner])
            elif self.fromHome[0] == 'l':
                self.result.set_teamHome(
                    matches[self.fromHome[1]].result.loser, ranks[matches[self.fromHome[1]].result.loser])
            elif self.fromHome[0] == 'p':
                self.result.set_teamHome(groups[self.fromHome[1]].position[self.fromHome[2]],
                                         ranks[groups[self.fromHome[1]].position[self.fromHome[2]]])

            if self.fromAway[0] == 'w':
                self.result.set_teamAway(
                    matches[self.fromAway[1]].result.winner, ranks[matches[self.fromAway[1]].result.winner])
            elif self.fromAway[0] == 'l':
                self.result.set_teamAway(
                    matches[self.fromAway[1]].result.loser, ranks[matches[self.fromAway[1]].result.loser])
            elif self.fromAway[0] == 'p':
                self.result.set_teamAway(groups[self.fromAway[1]].position[self.fromAway[2]],
                                         ranks[groups[self.fromAway[1]].position[self.fromAway[2]]])

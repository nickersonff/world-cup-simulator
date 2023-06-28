from Group import Group
from Tournament import Tornment
from Match import Match
from FormatReader import FormatReader
import itertools as iter


class WorldCup(Tornment):

    def group_func(self, groups, matches, ranks, draw):
        g = self.nteams//self.ngroup  # ex: 48 e 4 = 12
        countM = 0
        matid = 0
        for ii in range(g):
            group = range(ii*self.ngroup, (ii+1)*self.ngroup)
            mat = list(iter.combinations(group, 2))
            countM = len(mat)
            for m in mat:
                mmm = Match(matid, ('p', 0, m[0]), ('p', 0, m[1]))
                mmm.setup(groups=groups, ranks=ranks)
                mmm.result.play()
                matches.append(mmm)
                matid += 1
        print(matid)
        self.compute_groups(groups, matches, draw)

    def compute_groups(self, groups, matches, draw):
        g = self.nteams//self.ngroup  # ex: 48 e 4 = 12
        match_group_3rd = []
        teams_group_3rd = []
        # make the group
        for ii in range(g):
            group = range(ii*self.ngroup, (ii+1)*self.ngroup)
            mat = list(iter.combinations(group, 2))
            countM = len(mat)
            matches_group = range(ii*countM, (ii+1)*countM)
            teams_group = [draw.position[kk]
                           for kk in range(ii*self.ngroup, ii*self.ngroup+self.ngroup)]
            ggg = Group(teams_group, [matches[ii] for ii in matches_group])
            ggg.compute()
            match_group_3rd.extend(matches_group)
            teams_group_3rd.append(ggg.position[2])
            groups.append(ggg)

        best_3rdA = Group(teams_group_3rd, [matches[ii]
                                            for ii in match_group_3rd])
        best_3rdA.compute()
        groups.append(best_3rdA)

    def get_matches_group(self, roundid=0):
        form = FormatReader()
        rounds = form.getRounds(self.format)
        mat_per_group = dict()
        for r in range(roundid):
            for m in rounds[r].findall("match"):
                g = int(m.attrib['teamHome'])//self.ngroup
                if g not in mat_per_group:
                    mat_per_group[g] = [int(m.attrib['id'])]
                else:
                    mat_per_group[g].append(int(m.attrib['id']))

        return mat_per_group

    def compute(self, roundid, groups, matches, draw):
        g = self.nteams//self.ngroup  # ex: 48 e 4 = 12
        match_group_3rd = []
        teams_group_3rd = []
        # make the group
        matches_per_group = self.get_matches_group(roundid=roundid)
        for ii in range(g):
            matches_group = matches_per_group[ii]
            teams_group = [draw.position[kk]
                           for kk in range(ii*self.ngroup, ii*self.ngroup+self.ngroup)]
            ggg = Group(teams_group, [matches[ii] for ii in matches_group])
            ggg.compute()
            match_group_3rd.extend(matches_group)
            teams_group_3rd.append(ggg.position[2])
            groups.append(ggg)

        best_3rdA = Group(teams_group_3rd, [matches[ii]
                                            for ii in match_group_3rd])
        best_3rdA.compute()
        groups.append(best_3rdA)

    def double_elimination(self, groups, matches, ranks, draw):
        


    def run(self, rankgain=0, format="Cup48_4groups"):

        draw = Group(self.teams_list)
        f = FormatReader()
        inf = f.getTeamsGroups(format)
        # index 0 - number of teams / index 1 - number of teams per group
        self.ngroup = int(inf[1])
        self.nteams = int(inf[0])
        self.format = format
        print(self.ngroup, self.nteams)
        draw.draw()
        groups = [draw]
        if self.ranks == None:
            ranks = {}
            for ii in range(len(self.teams_list)):
                ranks[self.teams_list[ii]] = ii
        else:
            ranks = self.ranks

        gstart = [0]
        matches = []

        for r in f.getRounds(format):
            func = ""
            try:
                func = r.attrib['func']
            except:
                func = "NoNe"
            if func == "GroupPhase":
                # round 1 to round 3 moved to function
                # index 0 - number of teams / index 1 - number of teams per group
                self.group_func(groups, matches, ranks, draw)
            elif func == "DoubleElimination":
                self.double_elimination(groups, matches, ranks, draw)
            else:
                if func == "compute":
                    self.compute(int(r.attrib['id']), groups, matches, draw)
                for m in r.findall("match"):
                    if m.attrib['typeHome'] == 'p':
                        matches.append(Match(int(m.attrib['id']),
                                             (m.attrib['typeHome'], int(m.attrib['groupHome']),
                                              int(m.attrib['teamHome'])),
                                             (m.attrib['typeAway'], int(m.attrib['groupAway']),
                                              int(m.attrib['teamAway']))))
                    else:
                        matches.append(Match(int(m.attrib['id']),
                                             (m.attrib['typeHome'],
                                              int(m.attrib['matchHome'])),
                                             (m.attrib['typeAway'], int(m.attrib['matchAway']))))

                    matches[int(m.attrib['id'])].setup(
                        matches=matches, ranks=ranks, groups=groups)
                    matches[int(m.attrib['id'])].result.play(gain=rankgain)

                gstart.append(len(matches))

        print(groups[1].position)
        self.matches = matches
        self.ranks = ranks

    def classify(self):  # alterar ??? #

        self.classification = ['' for ii in self.teams_list]
        self.classification[0] = self.matches[-1].result.winner
        self.classification[1] = self.matches[-1].result.loser
        self.classification[2] = self.matches[-2].result.winner
        self.classification[3] = self.matches[-2].result.loser

        mm = 4
# Autria duplicada
        myteams = [self.matches[ii].result.loser for ii in range(96, 100)]
        mygroup = Group(myteams, self.matches)
        mygroup.compute()
        for ii in range(len(myteams)):
            self.classification[mm+ii] = mygroup.position[ii]

        mm = mm + len(myteams)

        myteams = [self.matches[ii].result.loser for ii in range(88, 96)]
        mygroup = Group(myteams, self.matches)
        mygroup.compute()
        for ii in range(len(myteams)):
            self.classification[mm+ii] = mygroup.position[ii]

        mm = mm + len(myteams)

        myteams = [self.matches[ii].result.loser for ii in range(72, 88)]
        mygroup = Group(myteams, self.matches)
        mygroup.compute()
        for ii in range(len(myteams)):
            self.classification[mm+ii] = mygroup.position[ii]

        mm = mm + len(myteams)

        inteams = [self.classification[ii] for ii in range(0, mm)]
        myteams = [ii for ii in self.teams_list if ii not in inteams]

        mygroup = Group(myteams, self.matches)
        mygroup.compute()
        for ii in range(len(myteams)):
            self.classification[mm+ii] = mygroup.position[ii]

    def print_matches(self):  # errado, somente para o de grupo de 4
        print('1st Round')
        [print(ii.result.teams)
         for iix, ii in enumerate(self.matches) if iix in range(0, 24)]
        print('2nd Round')
        [print(ii.result.teams)
         for iix, ii in enumerate(self.matches) if iix in range(24, 48)]
        print('3rd Round')
        [print(ii.result.teams)
         for iix, ii in enumerate(self.matches) if iix in range(48, 72)]
        print('Round of 32')
        [print(ii.result.teams)
         for iix, ii in enumerate(self.matches) if iix in range(72, 88)]
        print('Round of 16')
        [print(ii.result.teams)
         for iix, ii in enumerate(self.matches) if iix in range(88, 96)]
        print('Quarters')
        [print(ii.result.teams)
         for iix, ii in enumerate(self.matches) if iix in range(96, 100)]
        print('Semi-Final')
        [print(ii.result.teams)
         for iix, ii in enumerate(self.matches) if iix in range(100, 102)]
        print('3rd Place')
        [print(ii.result.teams)
         for iix, ii in enumerate(self.matches) if iix in range(102, 103)]
        print('Final')
        [print(ii.result.teams)
         for iix, ii in enumerate(self.matches) if iix in range(103, 104)]

    def print_matches3(self):
        print('1st Round')
        [print(ii.result.teams)
         for iix, ii in enumerate(self.matches) if iix in range(0, 16)]
        print('2nd Round')
        [print(ii.result.teams)
         for iix, ii in enumerate(self.matches) if iix in range(16, 32)]
        print('3rd Round')
        [print(ii.result.teams)
         for iix, ii in enumerate(self.matches) if iix in range(32, 48)]
        print('Round of 32')
        [print(ii.result.teams)
         for iix, ii in enumerate(self.matches) if iix in range(48, 64)]
        print('Round of 16')
        [print(ii.result.teams)
         for iix, ii in enumerate(self.matches) if iix in range(64, 72)]
        print('Quarters')
        [print(ii.result.teams)
         for iix, ii in enumerate(self.matches) if iix in range(72, 76)]
        print('Semi-Final')
        [print(ii.result.teams)
         for iix, ii in enumerate(self.matches) if iix in range(76, 78)]
        print('3rd Place')
        [print(ii.result.teams)
         for iix, ii in enumerate(self.matches) if iix in range(78, 79)]
        print('Final')
        [print(ii.result.teams)
         for iix, ii in enumerate(self.matches) if iix in range(79, 80)]

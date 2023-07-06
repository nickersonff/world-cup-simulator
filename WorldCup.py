from Group import Group
from Tournament import Tornment
from Match import Match
from FormatReader import FormatReader
import itertools as iter
from Cup48 import Cup48


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

    def double_elimination(self, groups, matches, ranks):
        #g = self.nteams//self.ngroup  # ex: 48 e 4 = 12
        #matid = 0
        #for ii in range(g):
        #    grp = range(ii*self.ngroup, (ii+1)*self.ngroup)
        #    for m in grp:
        #        if m<len(grp):
        #            mmm = Match(matid, ('p', 0, m*g), ('p', 0, m*g+1))
        #            mmm.setup(groups=groups, ranks=ranks)
        #            mmm.result.play()
        #            matches.append(mmm)
        #           matid += 1
        cup = Cup48(self.teams_list, self.ranks)
        cup.run()
        return cup

    def run(self, rankgain=0, format="Cup48_4groups"):

        draw = Group(self.teams_list)
        f = FormatReader()
        inf = f.getTeamsGroups(format)
        # index 0 - number of teams / index 1 - number of teams per group
        self.ngroup = int(inf[1])
        self.nteams = int(inf[0])
        self.format = format
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
                ccc = self.double_elimination(groups, matches, ranks)
                matches = ccc.matches
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

        self.matches = matches
        self.ranks = ranks


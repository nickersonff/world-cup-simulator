from Group import Group
from Tournament import Tornment
from Match import Match


class Cup48(Tornment):

    def run(self, rankgain=0):
        draw = Group(self.teams_list)
        # draw.draw4()
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
        # Round 1
        for ii in range(24):
            mmm = Match(gstart[0]+ii, ('p', 0, ii*2), ('p', 0, ii*2+1))
            mmm.setup(groups=groups, ranks=ranks)
            mmm.result.play(gain=rankgain)
            matches.append(mmm)
        gstart.append(len(matches))

        # Round 2
        for ii in range(12):
            mmm = Match(gstart[1]+ii, ('w', ii*2), ('w', ii*2+1))
            mmm.setup(matches=matches, ranks=ranks)
            mmm.result.play(gain=rankgain)
            matches.append(mmm)
        gstart.append(len(matches))

        # losers Round 1 - wild card
        for ii in range(12):
            mmm = Match(gstart[2]+ii, ('l', ii*2), ('l', ii*2+1))
            mmm.setup(matches=matches, ranks=ranks)
            mmm.result.play(gain=rankgain)
            matches.append(mmm)
        gstart.append(len(matches))

        # Round 3
        for ii in range(6):
            mmm = Match(gstart[3]+ii, ('w', gstart[1]+ii*2),
                        ('w', gstart[1]+ii*2+1))
            mmm.setup(matches=matches, ranks=ranks)
            mmm.result.play(gain=rankgain)
            matches.append(mmm)
        gstart.append(len(matches))

        # losers Round 2 - wild card
        xii = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
        for ii in range(12):
            mmm = Match(gstart[4]+ii, ('w', gstart[2]+ii),
                        ('l', gstart[1]+xii[ii]))
            mmm.setup(matches=matches, ranks=ranks)
            mmm.result.play(gain=rankgain)
            matches.append(mmm)
        gstart.append(len(matches))

        # make the group
        matches_group1 = []
        matches_group1.extend([gstart[0]+ii for ii in range(12)])
        matches_group1.extend([gstart[1]+ii for ii in range(6)])
        matches_group1.extend([gstart[2]+ii for ii in range(6)])
        matches_group1.extend([gstart[3]+ii for ii in range(3)])
        matches_group1.extend([gstart[4]+ii for ii in range(6)])

        teams_group1 = []
        teams_group1.extend(
            [matches[gstart[3]+ii].result.loser for ii in range(3)])
        teams_group1.extend(
            [matches[gstart[4]+ii].result.winner for ii in range(6)])

        # make the group
        matches_group2 = []
        matches_group2.extend([gstart[0]+ii for ii in range(12, 24)])
        matches_group2.extend([gstart[1]+ii for ii in range(6, 12)])
        matches_group2.extend([gstart[2]+ii for ii in range(6, 12)])
        matches_group2.extend([gstart[3]+ii for ii in range(3, 6)])
        matches_group2.extend([gstart[4]+ii for ii in range(6, 12)])

        teams_group2 = []
        teams_group2.extend(
            [matches[gstart[3]+ii].result.loser for ii in range(3, 6)])
        teams_group2.extend(
            [matches[gstart[4]+ii].result.winner for ii in range(6, 12)])

        group1 = Group(teams_group1, [matches[ii] for ii in matches_group1])
        group2 = Group(teams_group2, [matches[ii] for ii in matches_group2])
        group1.compute()
        group2.compute()

        groups.append(group1)
        groups.append(group2)

        # Round of 16
        matches.append(Match(66, ('w', 48), ('w', 49)))
        matches.append(Match(67, ('w', 50), ('p', 1, 0)))
        matches.append(Match(68, ('w', 51), ('w', 52)))
        matches.append(Match(69, ('w', 53), ('p', 2, 0)))
        gstart.append(len(matches))

        # losers round 3 - wild card
        matches.append(Match(70, ('p', 1, 1), ('p', 1, 8)))
        matches.append(Match(71, ('p', 1, 2), ('p', 1, 7)))
        matches.append(Match(72, ('p', 1, 3), ('p', 1, 6)))
        matches.append(Match(73, ('p', 1, 4), ('p', 1, 5)))

        matches.append(Match(74, ('p', 2, 1), ('p', 2, 8)))
        matches.append(Match(75, ('p', 2, 2), ('p', 2, 7)))
        matches.append(Match(76, ('p', 2, 3), ('p', 2, 6)))
        matches.append(Match(77, ('p', 2, 4), ('p', 2, 5)))
        gstart.append(len(matches))

        # Quarters
        matches.append(Match(78, ('w', 66), ('w', 67)))
        matches.append(Match(79, ('w', 68), ('w', 69)))
        gstart.append(len(matches))

        # losers round of 16 - wild card
        matches.append(Match(80, ('w', 70), ('l', 68)))
        matches.append(Match(81, ('w', 71), ('l', 69)))
        matches.append(Match(82, ('w', 72), ('w', 73)))
        matches.append(Match(83, ('w', 74), ('l', 66)))
        matches.append(Match(84, ('w', 75), ('l', 67)))
        matches.append(Match(85, ('w', 76), ('w', 77)))
        gstart.append(len(matches))

        # losers Quarters - wild card
        matches.append(Match(86, ('w', 80), ('w', 81)))
        matches.append(Match(87, ('w', 82), ('l', 79)))
        matches.append(Match(88, ('w', 83), ('w', 84)))
        matches.append(Match(89, ('w', 85), ('l', 78)))
        gstart.append(len(matches))

        # Quarters - wild card
        matches.append(Match(90, ('w', 86), ('w', 87)))
        matches.append(Match(91, ('w', 88), ('w', 89)))
        gstart.append(len(matches))

        # semi-final
        matches.append(Match(92, ('w', 78), ('w', 91)))
        matches.append(Match(93, ('w', 79), ('w', 90)))
        gstart.append(len(matches))

        # 3rd
        matches.append(Match(94, ('l', 92), ('l', 93)))
        gstart.append(len(matches))

        # final
        matches.append(Match(95, ('w', 92), ('w', 93)))
        gstart.append(len(matches))

        for mm in range(66, 96):
            matches[mm].setup(matches=matches, groups=groups, ranks=ranks)
            matches[mm].result.play(gain=rankgain)

        self.matches = matches
        self.ranks = ranks

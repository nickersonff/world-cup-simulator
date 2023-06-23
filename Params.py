import pandas as pd
import numpy as np
from FormatReader import FormatReader


class Params:

    BETA = 3.7581
    ALFA = 2.5156
    MIN_RATING = 1667
    MAX_RATING = 2143
    teams_list = ['Argentina',
                  'Brazil',
                  'France',
                  'Netherlands',
                  'Portugal',
                  'Spain',
                  'Italy',
                  'England',
                  'Germany',
                  'Croatia',
                  'Belgium',
                  'Colombia',
                  'Uruguay',
                  'Denmark',
                  'Switzerland',
                  'Morocco',
                  'Peru',
                  'Japan',
                  'Ecuador',
                  'Serbia',
                  'Hungary',
                  'Ukraine',
                  'United States']
    ranklist = {}

    def load_elo_rating(format="Cup48_4groups"):
        elo = pd.read_csv('./content/Elo-Rating-2022.csv')
        form = FormatReader()
        inf = form.getTeamsGroups(format)
        # index 0 - number of teams / index 1 - number of teams per group
        ngroup = int(inf[1])
        nteams = int(inf[0])
        Params.teams_list = elo['Team'][0:nteams]
        Params.MIN_RATING = np.min(elo['Rating'][0:nteams])
        Params.MAX_RATING = np.max(elo['Rating'][0:nteams])

        for rankval, team in enumerate(Params.teams_list):
            norm = 1 + np.exp(Params.BETA) * (elo[elo['Team'] == team]
                                              ['Rating'] - Params.MIN_RATING) / (Params.MAX_RATING - Params.MIN_RATING)
            Params.ranklist[team] = {'rank': rankval+1,
                                     'elo-rating': float(norm)}

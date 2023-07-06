import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Params import Params

df = pd.read_csv('./content/WorldCupMatches.csv')
do = pd.read_csv('./content/Elo-Rating.csv', encoding='latin-1')
toto = df[(df['Year'] > 1991) & (df['Year'] < 2022)]

countries = list(set(do['Team']))
years = np.arange(1994, 2015, 4)
ranks = {}

for yy in years:
    ranks[yy] = {}
    doy = do[(do['Year'] == yy)]

    ccin = list(set(doy['Team']))
    for cc in countries:
        if cc in ccin:
            ranks[yy][cc] = list(doy[doy['Team'] == cc]['Rating'])[0]
            if (ranks[yy][cc] < Params.MIN_RATING):
                ranks[yy][cc] = Params.MIN_RATING
        else:
            ranks[yy][cc] = Params.MIN_RATING

dif_rank = []
dif_goal = []
for yy in np.arange(1994, 2015, 4):
    hteams = list(toto[toto['Year'] == float(yy)]['Home Team Name'])
    ateams = list(toto[toto['Year'] == float(yy)]['Away Team Name'])
    hgoal = list(toto[toto['Year'] == float(yy)]['Home Team Goals'])
    agoal = list(toto[toto['Year'] == float(yy)]['Away Team Goals'])
    for hh in hteams:
        if hh not in ranks[yy]:
            ranks[yy][hh] = Params.MIN_RATING
    for hh in ateams:
        if hh not in ranks[yy]:
            ranks[yy][hh] = Params.MIN_RATING
    hteams_rank = [ranks[yy][hh] for hh in hteams]
    ateams_rank = [ranks[yy][hh] for hh in ateams]
    dif_rank.extend([hh-aa for hh, aa in zip(hteams_rank, ateams_rank)])
    dif_goal.extend([hh-aa for hh, aa in zip(hgoal, agoal)])

for ii in range(len(dif_rank)):
    dif_goal[ii] = -1 * dif_goal[ii]
    if (dif_rank[ii] < 0):
        dif_rank[ii] = -1 * dif_rank[ii]
        dif_goal[ii] = -1 * dif_goal[ii]

plt.plot(dif_rank, dif_goal, 'kx')

plt.hist(dif_rank, bins=np.arange(0, 50, 5))
plt.show()
from FormatReader import FormatReader
from WorldCup import WorldCup
from Params import Params
import itertools as iter
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

Params.load_elo_rating(format="Cup48")
cup = WorldCup(teams_list=Params.teams_list, ranks=Params.ranklist)
cup.run(format="Cup48")

rankgain = 0.0
nn = 1000
modelA_high = [0 for ii in range(nn)]
modelB_high = [0 for ii in range(nn)]
modelC_high = [0 for ii in range(nn)]
print("running...")
for ii in range(nn):
    ccc = WorldCup(teams_list=Params.teams_list, ranks=Params.ranklist)
    ccc.run(format="Cup48")
    ttt = ccc.match_type_dist()
    modelA_high[ii] = ttt['high']

    ccc = WorldCup(teams_list=Params.teams_list, ranks=Params.ranklist)
    # ccc.run(rankgain,ranklist)
    ccc.run(format="Cup48_3groups")
    ttt = ccc.match_type_dist()
    modelB_high[ii] = ttt['high']

    ccc = WorldCup(teams_list=Params.teams_list, ranks=Params.ranklist)
    # ccc.run(rankgain,ranklist)
    ccc.run(format="Cup48_4groups")
    ttt = ccc.match_type_dist()
    modelC_high[ii] = ttt['high']


plt.hist(modelB_high, bins=np.arange(11)-0.5, color='k', alpha=0.5)
plt.hist(modelA_high, bins=np.arange(11)-0.5, color='r', alpha=0.5)
plt.hist(modelC_high, bins=np.arange(11)-0.5, color='b', alpha=0.5)
plt.show()
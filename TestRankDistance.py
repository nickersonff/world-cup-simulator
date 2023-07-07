#
#
#
from FormatReader import FormatReader
from WorldCup import WorldCup
from Params import Params
import itertools as iter
import matplotlib.pyplot as plt
import numpy as np

# use this function to load initial parameters from content path for specific cup format name
Params.load_elo_rating(format="Cup48")

nn = 1000
modelA_high = [0 for ii in range(nn)]
modelB_high = [0 for ii in range(nn)]
modelC_high = [0 for ii in range(nn)]

for ii in range(nn):
    # instantiate the WorldCup class to run simulation in particular cup format
    ccc = WorldCup(teams_list=Params.teams_list, ranks=Params.ranklist)
    ccc.run(format="Cup48")
    ttt = ccc.match_type_dist()
    modelA_high[ii] = ttt['high']

    # diferent formats can be used to compare the results
    ccc = WorldCup(teams_list=Params.teams_list, ranks=Params.ranklist)
    ccc.run(format="Cup48_3groups")
    ttt = ccc.match_type_dist()
    modelB_high[ii] = ttt['high']

    ccc = WorldCup(teams_list=Params.teams_list, ranks=Params.ranklist)
    ccc.run(format="Cup48_4groups")
    ttt = ccc.match_type_dist()
    modelC_high[ii] = ttt['high']


plt.hist(modelB_high, bins=np.arange(11)-0.5, color='k', alpha=0.5)
plt.hist(modelA_high, bins=np.arange(11)-0.5, color='r', alpha=0.5)
plt.hist(modelC_high, bins=np.arange(11)-0.5, color='b', alpha=0.5)
plt.show()

from sklearn.neighbors import KernelDensity
from FormatReader import FormatReader
from WorldCup import WorldCup
from Params import Params
import itertools as iter
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

Params.load_elo_rating(format="Cup48")

nn = 2500
modelA_high = [0 for ii in range(nn)]
modelB_high = [0 for ii in range(nn)]
modelC_high = [0 for ii in range(nn)]
modelA_reg = [0 for ii in range(nn)]
modelB_reg = [0 for ii in range(nn)]
modelC_reg = [0 for ii in range(nn)]
modelA_low = [0 for ii in range(nn)]
modelB_low = [0 for ii in range(nn)]
modelC_low = [0 for ii in range(nn)]

for ii in range(nn):
    ccc = WorldCup(teams_list=Params.teams_list, ranks=Params.ranklist)
    ccc.run(format="Cup48")
    ttt = ccc.match_type_dist()
    modelA_high[ii] = ttt['high']
    modelA_reg[ii] = ttt['reg']
    modelA_low[ii] = ttt['low']

    ccc = WorldCup(teams_list=Params.teams_list, ranks=Params.ranklist)
    ccc.run(format="Cup48_3groups")
    ttt = ccc.match_type_dist()
    modelB_high[ii] = ttt['high']
    modelB_reg[ii] = ttt['reg']
    modelB_low[ii] = ttt['low']

    ccc = WorldCup(teams_list=Params.teams_list, ranks=Params.ranklist)
    ccc.run(format="Cup48_4groups")
    ttt = ccc.match_type_dist()
    modelC_high[ii] = ttt['high']
    modelC_reg[ii] = ttt['reg']
    modelC_low[ii] = ttt['low']


plt.hist(modelB_high, bins=np.arange(20)-0.5, color='k', alpha=0.5)
plt.hist(modelA_high, bins=np.arange(20)-0.5, color='r', alpha=0.5)
plt.hist(modelC_high, bins=np.arange(20)-0.5, color='b', alpha=0.5)
plt.show()

plt.figure(figsize=(8, 4))

kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(
    np.array(modelA_high)[:, np.newaxis])
X_plot = np.linspace(0, 15, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens)*100, 'r', label='Double elimination')
kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(
    np.array(modelB_high)[:, np.newaxis])
X_plot = np.linspace(0, 15, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens)*100, 'k', label='Group of 3')
kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(
    np.array(modelC_high)[:, np.newaxis])
X_plot = np.linspace(0, 15, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens)*100, 'k--', label='Group of 4')

for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)

plt.xlabel('Number of Matches')
plt.ylabel('Probability (%)')
plt.title('Number of TOP matches (2 TOP8 teams)')

plt.xlim(-0.5, 10.5)
plt.legend()
plt.show()

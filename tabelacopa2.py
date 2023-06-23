import seaborn as sns
from sklearn.neighbors import KernelDensity
import pylab
import scipy.optimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Match import Match
from Group import Group
from Tournament import Tornment
from Params import Params


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
              'United States',
              'Mexico',
              'Poland',
              'South Korea',
              'Czechia',
              'Iran',
              'Australia',
              'Sweden',
              'Norway',
              'Russia',
              'Scotland',
              'Austria',
              'Tunisia',
              'Senegal',
              'Costa Rica',
              'Paraguay',
              'Wales',
              'Turkey',
              'Canada',
              'Algeria',
              'Chile',
              'Greece',
              'Finland',
              'Venezuela',
              'Cameroon',
              'Ireland',
              'Ivory Coast',
              'Slovenia']

elo = pd.read_csv('./content/Elo-Rating-2022.csv')
ranklist = {}
for rankval, team in enumerate(teams_list):
    ranklist[team] = {'rank': rankval+1, 'elo-rating': 7}

df = pd.read_csv('./content/WorldCupMatches.csv')
# do = pd.read_csv('/content/fifa_ranking-2022-10-06.csv')
do = pd.read_csv('./content/Elo-Rating.csv', encoding='latin-1')
toto = df[(df['Year'] > 1991) & (df['Year'] < 2022)]


countries = list(set(do['Team']))
years = np.arange(1994, 2015, 4)
ranks = {}

""" ALTERAR A DIFERENÇA DO RANK ?? """

for yy in years:
    ranks[yy] = {}
    # doy = do[((do['rank_date']>str(yy)) & (do['rank_date']<str(yy+1)))]
    doy = do[(do['Year'] == yy)]

    ccin = list(set(doy['Team']))
    for cc in countries:
        if cc in ccin:
            ranks[yy][cc] = list(doy[doy['Team'] == cc]['Rating'])[0]
            if (ranks[yy][cc] < param.MIN_RATING):
                ranks[yy][cc] = param.MIN_RATING
        else:
            ranks[yy][cc] = param.MIN_RATING

dif_rank = []
dif_goal = []
for yy in np.arange(1994, 2015, 4):
    hteams = list(toto[toto['Year'] == float(yy)]['Home Team Name'])
    ateams = list(toto[toto['Year'] == float(yy)]['Away Team Name'])
    hgoal = list(toto[toto['Year'] == float(yy)]['Home Team Goals'])
    agoal = list(toto[toto['Year'] == float(yy)]['Away Team Goals'])
    for hh in hteams:
        if hh not in ranks[yy]:
            ranks[yy][hh] = param.MIN_RATING
    for hh in ateams:
        if hh not in ranks[yy]:
            ranks[yy][hh] = param.MIN_RATING
    hteams_rank = [ranks[yy][hh] for hh in hteams]
    ateams_rank = [ranks[yy][hh] for hh in ateams]
    dif_rank.extend([hh-aa for hh, aa in zip(hteams_rank, ateams_rank)])
    dif_goal.extend([hh-aa for hh, aa in zip(hgoal, agoal)])

""" FIM DA ALTERAÇÃO """


for ii in range(len(dif_rank)):
    dif_goal[ii] = -1 * dif_goal[ii]
    if (dif_rank[ii] < 0):
        dif_rank[ii] = -1 * dif_rank[ii]
        dif_goal[ii] = -1 * dif_goal[ii]

plt.plot(dif_rank, dif_goal, 'kx')

plt.hist(dif_rank, bins=np.arange(0, 50, 5))

xxx = np.arange(-4, 7200, 5)
xxx[0] = 0

avictory = [0 for ii in xxx]
bvictory = [0 for ii in xxx]
ddraw = [0 for ii in xxx]
fgames = [0 for ii in xxx]

for dr, dg in zip(dif_rank, dif_goal):
    print(dr)
    iix = int(dr/4)
    print(iix)
    if (dr != 0):
        iix = iix+1
    if (dg > 0):
        avictory[iix] += 1
    if (dg < 0):
        bvictory[iix] += 1
    if (dg == 0):
        ddraw[iix] += 1
    fgames[iix] += 1

for ii in range(len(fgames)):
    if (fgames[ii] == 0):
        fgames[ii] = 1

aaa = [aa/bb for aa, bb in zip(avictory, fgames)]
bbb = [aa/bb for aa, bb in zip(bvictory, fgames)]
ccc = [aa/bb for aa, bb in zip(ddraw, fgames)]

ccc[0] = np.sum(ddraw)/np.sum(fgames)
aaa[0] = (1-ccc[0])/2
bbb[0] = (1-ccc[0])/2


def parabola(x, a, b, c):
    return a*x**2 + b*x + c


zzz = np.polyfit(xxx[:10], aaa[:10], 1)
ppp = np.poly1d(zzz)
plt.plot(np.arange(70), ppp(np.arange(70)), 'r-')

zzz = np.polyfit(xxx[:10], bbb[:10], 1)
ppp = np.poly1d(zzz)
plt.plot(np.arange(70), ppp(np.arange(70)), 'k-')

zzz = np.polyfit(xxx[:15], ccc[:15], 1)
ppp = np.poly1d(zzz)
plt.plot(np.arange(70), ppp(np.arange(70)), 'g-')

plt.plot(xxx, aaa, 'rx')
plt.plot(xxx, bbb, 'kx')
plt.plot(xxx, ccc, 'gx')

ppp = 10000
zzz = []
ddd = []
eee = []
for ii in np.arange(0, 1.5, 0.1):
    ddf = np.random.poisson(1.5+ii, size=ppp) - \
        np.random.poisson(1.5-ii, size=ppp)
    zzz.append(np.sum(ddf > 0)/ppp)
    ddd.append(np.sum(ddf == 0)/ppp)
    eee.append(np.sum(ddf < 0)/ppp)

ooo = 50*np.arange(len(np.arange(0, 1.5, 0.1)))/len(np.arange(0, 1.5, 0.1))

plt.plot(ooo, 100*np.array(zzz))
plt.plot(ooo, 100*np.array(ddd))
plt.plot(ooo, 100*np.array(eee))

ooo = 100*np.arange(len(np.arange(0, 1.5, 0.1)))/len(np.arange(0, 1.5, 0.1))

plt.plot(xxx, 100*np.array(aaa), 'rx')
plt.plot(xxx, 100*np.array(bbb), 'bx')
plt.plot(xxx, 100*np.array(ccc), 'gx')

plt.plot(ooo, 100*np.array(zzz), 'r', label='High Rank Win')
plt.plot(ooo, 100*np.array(ddd), 'b', label='Draw')
plt.plot(ooo, 100*np.array(eee), 'g', label='Low Rank Win')

for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)

plt.xlabel('FIFA Rank Diference')
plt.ylabel('Frequency of Events')
plt.title('Match Results vs FIFA Rank Diference')
plt.xlim(-3, 65)
plt.legend()
plt.show()

"""Rank value index"""


ranks = np.zeros((50, 50))

for ii in range(50):
    for jj in range(ii, 50):
        ranks[ii, jj] = np.sqrt((1-(ii/50)) * (1-(jj/50)))
        ranks[jj, ii] = np.sqrt((1-(ii/50)) * (1-(jj/50)))


# plt.imshow(ranks, extent=[0, 1, 0, 1])

# for ii in [0,1,2,3,4,5,6,7,9,19,29,39,49]:
#  plt.plot(range(1,51),ranks[ii,:])


plt.figure(figsize=(2, 4))

XPoints = []
YPoints = []

for val in np.arange(1, 51, 1):
    XPoints.append(val)
    YPoints.append(val)

ZPoints = np.ndarray((len(XPoints), len(YPoints)))

for x in range(0, len(XPoints)):
    for y in range(0, len(YPoints)):
        ZPoints[x][y] = np.sqrt((1-((XPoints[x]-1)/50))
                                * (1-((YPoints[y]-1)/50)))*100

# Print x,y and z values
# print(XPoints)
# print(YPoints)
# print(ZPoints)

# Set the x axis and y axis limits
pylab.xlim([1, 50])
pylab.ylim([1, 50])

# Provide a title for the contour plot
plt.title('Match Rank Index')

# Set x axis label for the contour plot
plt.xlabel('FIFA Rank (Team A)')

# Set y axis label for the contour plot
plt.ylabel('FIFA Rank (Team B)')

# Create contour lines or level curves using matplotlib.pyplot module
contours = plt.contour(XPoints, YPoints, ZPoints, levels=[30, 45, 60, 75, 90])

# Display z values on contour lines
plt.clabel(contours, inline=1, fontsize=12, fmt='%1.0f')

for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)

# Display the contour plot
plt.show()

totalk = 50000
vals = [0 for ii in range(totalk)]
for ii in range(totalk):
    vals[ii] = np.sqrt((1-(np.random.randint(48)/50))
                       * (1-(np.random.randint(48)/50)))
# plt.hist(vals)

# plt.figure(figsize=(8, 4))

kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(
    np.array(vals)[:, np.newaxis])
X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens)*100, 'r', label='Repechage')

totalk = 50000
vals = [0 for ii in range(totalk)]
for ii in range(totalk):
    vals[ii] = np.sqrt((1-(np.random.randint(48)/50))
                       * (1-(np.random.randint(48)/50)))
plt.hist(vals, bins=np.arange(0, 1, 0.01))


"""# Rank Distribution Analysis"""

rankgain = 0.0
nn = 1000
modelA_high = [0 for ii in range(nn)]
modelB_high = [0 for ii in range(nn)]
modelC_high = [0 for ii in range(nn)]
for ii in range(nn):
    ccc = Cup48(teams_list, ranklist)
    ccc.run(rankgain)
    ttt = ccc.match_type_dist()
    modelA_high[ii] = ttt['high']

    ccc = Cup48_3groups(teams_list)
    # ccc.run(rankgain,ranklist)
    ccc.run(rankgain)
    ttt = ccc.match_type_dist()
    modelB_high[ii] = ttt['high']

    ccc = Cup48_4groups(teams_list)
    # ccc.run(rankgain,ranklist)
    ccc.run(rankgain)
    ttt = ccc.match_type_dist()
    modelC_high[ii] = ttt['high']

plt.hist(modelB_high, bins=np.arange(11)-0.5, color='k', alpha=0.5)
plt.hist(modelA_high, bins=np.arange(11)-0.5, color='r', alpha=0.5)
plt.hist(modelC_high, bins=np.arange(11)-0.5, color='b', alpha=0.5)


# plt.hist(eee,bins=np.arange(11)-0.5,color='b',alpha=0.5)


# e por sorte?

rankgain = 0.7
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
    ccc = Cup48(teams_list, ranklist)
    ccc.run(rankgain)
    ttt = ccc.match_type_dist()
    modelA_high[ii] = ttt['high']
    modelA_reg[ii] = ttt['reg']
    modelA_low[ii] = ttt['low']

    ccc = Cup48_3groups(teams_list, ranklist)
    ccc.run(rankgain)
    ttt = ccc.match_type_dist()
    modelB_high[ii] = ttt['high']
    modelB_reg[ii] = ttt['reg']
    modelB_low[ii] = ttt['low']

    ccc = Cup48_4groups(teams_list, ranklist)
    ccc.run(rankgain)
    ttt = ccc.match_type_dist()
    modelC_high[ii] = ttt['high']
    modelC_reg[ii] = ttt['reg']
    modelC_low[ii] = ttt['low']

plt.hist(modelB_high, bins=np.arange(20)-0.5, color='k', alpha=0.5)
plt.hist(modelA_high, bins=np.arange(20)-0.5, color='r', alpha=0.5)
plt.hist(modelC_high, bins=np.arange(20)-0.5, color='b', alpha=0.5)
print(np.mean(modelA_high))
print(np.mean(modelB_high))
print(np.mean(modelC_high))
print(np.mean(modelA_high)/np.mean(modelB_high))
print(np.mean(modelA_high)/np.mean(modelC_high))

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

plt.hist(modelB_reg, bins=np.arange(16, 50)-0.5, color='k', alpha=0.5)
plt.hist(modelA_reg, bins=np.arange(16, 50)-0.5, color='r', alpha=0.5)
plt.hist(modelC_reg, bins=np.arange(16, 50)-0.5, color='b', alpha=0.5)
print(np.mean(modelA_reg))
print(np.mean(modelB_reg))
print(np.mean(modelC_reg))
print(np.mean(modelA_reg)/np.mean(modelB_reg))
print(np.mean(modelA_reg)/np.mean(modelC_reg))

kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(
    np.array(modelA_reg)[:, np.newaxis])
X_plot = np.linspace(15, 50, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens)*100, 'r', label='Double elimination')
kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(
    np.array(modelB_reg)[:, np.newaxis])
X_plot = np.linspace(15, 50, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens)*100, 'k', label='Group of 3')
kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(
    np.array(modelC_reg)[:, np.newaxis])
X_plot = np.linspace(15, 50, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens)*100, 'k--', label='Group of 4')

for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)

plt.xlabel('Number of Matches')
plt.ylabel('Probability (%)')
plt.title('Number of HIGH matches (1 TOP8 teams)')
plt.xlim(15, 50)
plt.legend()
plt.show()

plt.hist(modelB_low, bins=np.arange(40, 80)-0.5, color='k', alpha=0.5)
plt.hist(modelA_low, bins=np.arange(40, 80)-0.5, color='r', alpha=0.5)
plt.hist(modelC_low, bins=np.arange(40, 80)-0.5, color='b', alpha=0.5)
print(np.mean(modelA_low))
print(np.mean(modelB_low))
print(np.mean(modelC_low))
print(np.mean(modelA_low)/np.mean(modelB_low))
print(np.mean(modelA_low)/np.mean(modelC_low))

kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(
    np.array(modelA_low)[:, np.newaxis])
X_plot = np.linspace(40, 80, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens)*100, 'r', label='Double elimination')
kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(
    np.array(modelB_low)[:, np.newaxis])
X_plot = np.linspace(40, 80, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens)*100, 'k', label='Group of 3')
kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(
    np.array(modelC_low)[:, np.newaxis])
X_plot = np.linspace(40, 80, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens)*100, 'k--', label='Group of 4')

for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)

plt.xlabel('Number of Matches')
plt.ylabel('Probability (%)')
plt.title('Number of REGULAR matches (0 TOP8 teams)')
plt.xlim(40, 80)
plt.legend()
plt.show()

plt.figure(figsize=(12, 4))

kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(
    np.array(modelA_low)[:, np.newaxis])
X_plot = np.linspace(40, 80, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens)*100, 'r', label='Repechage')
kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(
    np.array(modelB_low)[:, np.newaxis])
X_plot = np.linspace(40, 80, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens)*100, 'k', label='Group of 3')
kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(
    np.array(modelC_low)[:, np.newaxis])
X_plot = np.linspace(40, 80, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens)*100, 'k--', label='Group of 4')


kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(
    np.array(modelA_reg)[:, np.newaxis])
X_plot = np.linspace(15, 50, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens)*100, 'r')
kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(
    np.array(modelB_reg)[:, np.newaxis])
X_plot = np.linspace(15, 50, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens)*100, 'k')
kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(
    np.array(modelC_reg)[:, np.newaxis])
X_plot = np.linspace(15, 50, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens)*100, 'k--')

kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(
    np.array(modelA_high)[:, np.newaxis])
X_plot = np.linspace(0, 15, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens)*100, 'r', label='Repescagem')
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

"""# Rank Distribution Analysis B

# Nova seção
"""

rankgain = 0.7
nn = 500
kumo = 30000
modelA_rank = []
modelB_rank = []
modelC_rank = []
for ii in range(nn):
    ccc = Cup48(teams_list, ranklist)
    ccc.run(rankgain)
    ttt = ccc.match_rank_index()
    modelA_rank.extend(ttt)

    ccc = Cup48_3groups(teams_list, ranklist)
    ccc.run(rankgain)
    ttt = ccc.match_rank_index()
    modelB_rank.extend(ttt)

    ccc = Cup48_4groups(teams_list, ranklist)
    ccc.run(rankgain)
    ttt = ccc.match_rank_index()
    modelC_rank.extend(ttt)

modelA_rank = np.random.permutation(modelA_rank)[:kumo]
modelB_rank = np.random.permutation(modelB_rank)[:kumo]
modelC_rank = np.random.permutation(modelC_rank)[:kumo]

randon_rank = [0 for ii in range(kumo)]
for ii in range(kumo):
    rr1 = np.random.randint(48)
    rr2 = np.random.randint(48)
    while (rr2 == rr1):
        rr2 = np.random.randint(48)
    randon_rank[ii] = np.sqrt((1-(rr2/50))*(1-(rr1/50)))

plt.figure(figsize=(10, 4))

kde = KernelDensity(kernel="gaussian", bandwidth=0.12).fit(
    np.array(modelA_rank)[:, np.newaxis])
X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot*100, np.exp(log_dens), 'r', label='Double elimination')
kde = KernelDensity(kernel="gaussian", bandwidth=0.12).fit(
    np.array(modelB_rank)[:, np.newaxis])
X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot*100, np.exp(log_dens), 'k', label='Group of 3')
kde = KernelDensity(kernel="gaussian", bandwidth=0.12).fit(
    np.array(modelC_rank)[:, np.newaxis])
X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot*100, np.exp(log_dens), 'k--', label='Group of 4')
kde = KernelDensity(kernel="gaussian", bandwidth=0.12).fit(
    np.array(randon_rank)[:, np.newaxis])
X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot*100, np.exp(log_dens), 'k-', alpha=0.5, label='Random')


for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)

plt.xlabel('<- Low Ranked Matches -- **Rank Index** -- High Ranked Matches ->')
plt.ylabel('Normalized Frequency')
plt.title('Distribution of Matches by Rank Index')


plt.legend()
plt.show()

# plt.figure(figsize=(10, 4))
plt.figure(figsize=(5, 4))

mmm = np.min([modelA_rank, modelB_rank, modelC_rank])
mmm2 = np.max([modelA_rank, modelB_rank, modelC_rank])

kde = KernelDensity(kernel="gaussian", bandwidth=0.12).fit(
    np.array(randon_rank)[:, np.newaxis])
X_plot = np.linspace(mmm, mmm2, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
randome = np.exp(log_dens)

plt.plot([0, 100], [0, 0], 'k--', alpha=0.3, linewidth=0.8)

kde = KernelDensity(kernel="gaussian", bandwidth=0.12).fit(
    np.array(modelA_rank)[:, np.newaxis])
X_plot = np.linspace(mmm, mmm2, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot*100, 100*(np.exp(log_dens)/randome) -
         100, 'r', label='Double elimination')
kde = KernelDensity(kernel="gaussian", bandwidth=0.12).fit(
    np.array(modelB_rank)[:, np.newaxis])
X_plot = np.linspace(mmm, mmm2, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot*100, 100*(np.exp(log_dens)/randome) -
         100, 'k', label='Group of 3')
kde = KernelDensity(kernel="gaussian", bandwidth=0.12).fit(
    np.array(modelC_rank)[:, np.newaxis])
X_plot = np.linspace(mmm, mmm2, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot*100, 100*(np.exp(log_dens)/randome) -
         100, 'k--', label='Group of 4')

for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)

plt.xlabel('<- Low Ranked Matches    **Rank Index**    High Ranked Matches ->')
plt.ylabel('Frequency relative to random (%)')
plt.title('Frequency of Matches by Rank Index')

plt.legend()
plt.show()

totalk = 50000
vals = [0 for ii in range(totalk)]
for ii in range(totalk):
    vals[ii] = np.sqrt((1-(np.random.randint(48)/50))
                       * (1-(np.random.randint(48)/50)))
plt.hist(vals, bins=np.arange(0, 1, 0.01))
plt.hist(modelA_rank, bins=np.arange(0, 1, 0.01))

"""# Competitiveness"""

rankgain = 0.7
nn = 2500
# kumo = 15000
modelA_rank = []
modelB_rank = []
modelC_rank = []
for ii in range(nn):
    ccc = Cup48(teams_list)
    ccc.run(rankgain)
    ttt = ccc.match_rank_distance()
    modelA_rank.extend(ttt)

    ccc = Cup48_3groups(teams_list)
    ccc.run(rankgain)
    ttt = ccc.match_rank_distance()
    modelB_rank.extend(ttt)

    ccc = Cup48_4groups(teams_list)
    ccc.run(rankgain)
    ttt = ccc.match_rank_distance()
    modelC_rank.extend(ttt)

kumo = int(np.min([len(modelA_rank), len(modelB_rank), len(modelC_rank)])*.99)

modelA_rank = np.random.permutation(modelA_rank)[:kumo]
modelB_rank = np.random.permutation(modelB_rank)[:kumo]
modelC_rank = np.random.permutation(modelC_rank)[:kumo]

randon_rank = [0 for ii in range(kumo)]
for ii in range(kumo):
    rr1 = np.random.randint(48)
    rr2 = np.random.randint(48)
    while rr1 == rr2:
        rr2 = np.random.randint(48)
    randon_rank[ii] = np.abs(rr2-rr1)

# plt.figure(figsize=(10, 4))
plt.figure(figsize=(5, 4))

bww = 2

kde = KernelDensity(kernel="gaussian", bandwidth=bww).fit(
    np.array(randon_rank)[:, np.newaxis])
X_plot = np.linspace(1, 40, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
randome = np.exp(log_dens)

plt.plot([0, 40], [0, 0], 'k--', alpha=0.3, linewidth=0.8)

kde = KernelDensity(kernel="gaussian", bandwidth=bww).fit(
    np.array(modelA_rank)[:, np.newaxis])
X_plot = np.linspace(1, 40, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, 100*(np.exp(log_dens)/randome) -
         100, 'r', label='Double elimination')
kde = KernelDensity(kernel="gaussian", bandwidth=bww).fit(
    np.array(modelB_rank)[:, np.newaxis])
X_plot = np.linspace(1, 40, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, 100*(np.exp(log_dens)/randome)-100, 'k', label='Group of 3')
kde = KernelDensity(kernel="gaussian", bandwidth=bww).fit(
    np.array(modelC_rank)[:, np.newaxis])
X_plot = np.linspace(1, 40, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, 100*(np.exp(log_dens)/randome)-100, 'k--', label='Group of 4')


for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)

plt.xlabel('<- More competitive    **Rank Distance**    Less competitive ->')
plt.ylabel('Frequency relative to random (%)')
plt.title('Frequency of Matches by Rank Distance')


plt.legend()
plt.show()

plt.figure(figsize=(10, 4))

bww = 2

kde = KernelDensity(kernel="gaussian", bandwidth=bww).fit(
    np.array(randon_rank)[:, np.newaxis])
X_plot = np.linspace(1, 40, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
randome = np.exp(log_dens)
plt.plot(X_plot, randome, 'k', alpha=0.5, label='Random')

plt.plot([0, 40], [0, 0], 'k--', alpha=0.3, linewidth=0.8)

kde = KernelDensity(kernel="gaussian", bandwidth=bww).fit(
    np.array(modelA_rank)[:, np.newaxis])
X_plot = np.linspace(1, 40, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens), 'r', label='Double elimination')
kde = KernelDensity(kernel="gaussian", bandwidth=bww).fit(
    np.array(modelB_rank)[:, np.newaxis])
X_plot = np.linspace(1, 40, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens), 'k', label='Group of 3')
kde = KernelDensity(kernel="gaussian", bandwidth=bww).fit(
    np.array(modelC_rank)[:, np.newaxis])
X_plot = np.linspace(1, 40, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens), 'k--', label='Group of 4')


for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)

plt.xlabel('<- More competitive    **Rank Distance**    Less competitive ->')
plt.ylabel('Frequency relative to random (%)')
plt.title('Frequency of Matches by Rank Distance')

plt.legend()
plt.show()

"""#Fairness"""


def rank_fairness(ranklist, gamma=1):
    return np.sum([(np.abs(ii-iix)/len(ranklist))*(1-(ii/48))**gamma for iix, ii in enumerate(ranklist)])


totalruns = 1000

rankgain = 0.7

modelA_fairness = []
modelB_fairness = []
modelC_fairness = []
random_fairness = []


modelA_winner = []
modelB_winner = []
modelC_winner = []
random_winner = []

gmm = 10

for ii in range(totalruns):
    ccc = Cup48(teams_list, ranklist)
    ccc.run(rankgain)
    ccc.classify()
    modelA_fairness.append(rank_fairness(
        [ccc.ranks[nnn] for nnn in ccc.classification], gamma=gmm))
    modelA_winner.append(ccc.ranks[ccc.classification[0]])

    ccc = Cup48_3groups(teams_list, ranklist)
    ccc.run(rankgain)
    ccc.classify()
    modelB_fairness.append(rank_fairness(
        [ccc.ranks[nnn] for nnn in ccc.classification], gamma=gmm))
    modelB_winner.append(ccc.ranks[ccc.classification[0]])

    ccc = Cup48_4groups(teams_list, ranklist)
    ccc.run(rankgain)
    ccc.classify()
    modelC_fairness.append(rank_fairness(
        [ccc.ranks[nnn] for nnn in ccc.classification], gamma=gmm))
    modelC_winner.append(ccc.ranks[ccc.classification[0]])

    rrr = np.random.permutation(48)
    random_fairness.append(rank_fairness(rrr, gamma=gmm))
    random_winner.append(rrr[0])

plt.plot(np.arange(48), ((48-np.arange(48))/48)**40, 'k', alpha=0.5)

plt.figure(figsize=(10, 4))

bww = 0.12

mmax = np.max([modelA_fairness, modelB_fairness,
              modelC_fairness, random_fairness])
mmin = np.min([modelA_fairness, modelB_fairness,
              modelC_fairness, random_fairness])

kde = KernelDensity(kernel="gaussian", bandwidth=bww).fit(
    np.array(random_fairness)[:, np.newaxis])
X_plot = np.linspace(mmin, mmax, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
randome = np.exp(log_dens)

plt.plot([mmin, mmax], [0, 0], 'k--', alpha=0.3, linewidth=0.8)

kde = KernelDensity(kernel="gaussian", bandwidth=bww).fit(
    np.array(modelA_fairness)[:, np.newaxis])
X_plot = np.linspace(mmin, mmax, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens), 'r', label='Double elimination')
kde = KernelDensity(kernel="gaussian", bandwidth=bww).fit(
    np.array(modelB_fairness)[:, np.newaxis])
X_plot = np.linspace(mmin, mmax, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens), 'k', label='Group of 3')
kde = KernelDensity(kernel="gaussian", bandwidth=bww).fit(
    np.array(modelC_fairness)[:, np.newaxis])
X_plot = np.linspace(mmin, mmax, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens), 'k--', label='Group of 4')


for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)

plt.xlabel('<- More Fair    **Fairness Index**    Less Fair ->')
plt.ylabel('Frequency relative to random (%)')
plt.title('Frequency of Matches by Rank Distance')


plt.legend()
plt.show()

plt.figure(figsize=(10, 4))

bww = 2

mmax = np.max([modelA_winner, modelB_winner, modelC_winner, random_winner])
mmin = np.min([modelA_winner, modelB_winner, modelC_winner, random_winner])

kde = KernelDensity(kernel="gaussian", bandwidth=bww).fit(
    np.array(random_winner)[:, np.newaxis])
X_plot = np.linspace(mmin, mmax, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
randome = np.exp(log_dens)

plt.plot([mmin, mmax], [0, 0], 'k--', alpha=0.3, linewidth=0.8)

kde = KernelDensity(kernel="gaussian", bandwidth=bww).fit(
    np.array(modelA_winner)[:, np.newaxis])
X_plot = np.linspace(mmin, mmax, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens), 'r', label='Double elimination')
kde = KernelDensity(kernel="gaussian", bandwidth=bww).fit(
    np.array(modelB_winner)[:, np.newaxis])
X_plot = np.linspace(mmin, mmax, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens), 'k', label='Group of 3')
kde = KernelDensity(kernel="gaussian", bandwidth=bww).fit(
    np.array(modelC_winner)[:, np.newaxis])
X_plot = np.linspace(mmin, mmax, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_plot)
plt.plot(X_plot, np.exp(log_dens), 'k--', label='Group of 4')


for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)

plt.xlabel('<- More competitive    **Rank Distance**    Less competitive ->')
plt.ylabel('Frequency relative to random (%)')
plt.title('Frequency of Matches by Rank Distance')


plt.legend()
plt.show()


plt.figure(figsize=(5, 4))
mmax = np.max([modelA_fairness, modelB_fairness, modelC_fairness])
mmin = np.min([modelA_fairness, modelB_fairness, modelC_fairness])
bins_vals = np.linspace(mmin, mmax, 1000)

aaa, bbb = np.histogram(modelA_fairness, bins=bins_vals)
ccc = bbb[1:]-((bbb[1]-bbb[0])/2)
plt.plot(ccc, 100*np.cumsum(aaa)/np.sum(aaa), 'r', label='Double elimination')
aaa, bbb = np.histogram(modelB_fairness, bins=bins_vals)
ccc = bbb[1:]-((bbb[1]-bbb[0])/2)
plt.plot(ccc, 100*np.cumsum(aaa)/np.sum(aaa), 'k', label='Group of 3')
aaa, bbb = np.histogram(modelC_fairness, bins=bins_vals)
ccc = bbb[1:]-((bbb[1]-bbb[0])/2)
plt.plot(ccc, 100*np.cumsum(aaa)/np.sum(aaa), 'k--', label='Group of 4')
# aaa,bbb = np.histogram(random_fairness,bins=bins_vals)
# ccc = bbb[1:]-((bbb[1]-bbb[0])/2)
# plt.plot(ccc,100*np.cumsum(aaa)/np.sum(aaa),'k--',alpha=0.5,label='Random')

for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)

plt.xlabel('<- More Fair    **Fairness Index**    Less Fair ->')
plt.ylabel('Cumulative probability (%)')
plt.title('Cumulative probability of Tournament Fairness')

plt.xlim([0.4, 1.8])

plt.legend(loc='lower right')
plt.show()

plt.violinplot([modelA_fairness, modelB_fairness,
               modelC_fairness, random_fairness])

sns.violinplot(data=[modelA_winner, modelB_winner, modelC_winner])

ccc = Cup48_4groups(teams_list, ranklist)
ccc.run(rankgain)
ccc.print_matches()
ccc.classify()

"""# Schedulle Analysis"""

# cup-our

dependecies = {}
for ii in range(24):
    dependecies[ii] = []

for ii in range(12):
    dependecies[24+ii] = [ii*2, ii*2+1]
    dependecies[36+ii] = [ii*2, ii*2+1]

for ii in range(6):
    dependecies[48+ii] = [24+ii*2, 24+ii*2+1]
for ii in range(6):
    dependecies[54+ii*2] = [24+2*ii+1, 36+ii*2]
    dependecies[54+ii*2+1] = [24+2*ii, 36+ii*2+1]

dependecies[66] = [48, 49]
dependecies[67] = [50, 54, 55, 56, 57, 58, 59, 51, 52, 53]
dependecies[68] = [51, 52]
dependecies[69] = [53, 60, 61, 62, 63, 64, 65, 60, 61, 62]

dependecies[70] = [54, 55, 56, 57, 58, 59, 48, 49, 50]
dependecies[71] = [54, 55, 56, 57, 58, 59, 48, 49, 50]
dependecies[72] = [54, 55, 56, 57, 58, 59, 48, 49, 50]
dependecies[73] = [54, 55, 56, 57, 58, 59, 48, 49, 50]
dependecies[74] = [60, 61, 62, 63, 64, 65, 51, 52, 53]
dependecies[75] = [60, 61, 62, 63, 64, 65, 51, 52, 53]
dependecies[76] = [60, 61, 62, 63, 64, 65, 51, 52, 53]
dependecies[77] = [60, 61, 62, 63, 64, 65, 51, 52, 53]

dependecies[78] = [66, 67]
dependecies[79] = [68, 69]

dependecies[80] = [70, 66]
dependecies[81] = [71, 67]
dependecies[82] = [72, 73]
dependecies[83] = [74, 68]
dependecies[84] = [75, 69]
dependecies[85] = [76, 77]

dependecies[86] = [80, 81]
dependecies[87] = [82, 78]
dependecies[88] = [83, 84]
dependecies[89] = [85, 79]

dependecies[90] = [86, 87]
dependecies[91] = [88, 89]

dependecies[92] = [78, 91]
dependecies[93] = [79, 90]

dependecies[94] = [92, 93]
dependecies[95] = [92, 93]

dayless = [0 for ii in range(len(dependecies))]
for ii in range(36, 36+12):
    dayless[ii] = -1
for ii in range(54, 54+12):
    dayless[ii] = -1
for ii in range(70, 70+8):
    dayless[ii] = -1
for ii in range(80, 80+12):
    dayless[ii] = -1
dayless[94] = -1

dep = {'cup-our': dependecies}
deyl = {'cup-our': dayless}

# cup-our

dependecies = {}
for ii in range(24):
    dependecies[ii] = []

for ii in range(12):
    dependecies[24+ii] = [ii*2, ii*2+1]
    dependecies[36+ii] = [ii*2, ii*2+1]

for ii in range(6):
    dependecies[48+ii] = [24+ii*2, 24+ii*2+1]
for ii in range(6):
    dependecies[54+ii*2] = [24+2*ii+1, 36+ii*2]
    dependecies[54+ii*2+1] = [24+2*ii, 36+ii*2+1]

dependecies[66] = [48, 49]
dependecies[67] = [50, 54, 55, 56, 57, 58, 59, 51, 52, 53]
dependecies[68] = [51, 52]
dependecies[69] = [53, 60, 61, 62, 63, 64, 65, 60, 61, 62]

dependecies[70] = [54, 55, 56, 57, 58, 59, 48, 49, 50]
dependecies[71] = [54, 55, 56, 57, 58, 59, 48, 49, 50]
dependecies[72] = [54, 55, 56, 57, 58, 59, 48, 49, 50]
dependecies[73] = [54, 55, 56, 57, 58, 59, 48, 49, 50]
dependecies[74] = [60, 61, 62, 63, 64, 65, 51, 52, 53]
dependecies[75] = [60, 61, 62, 63, 64, 65, 51, 52, 53]
dependecies[76] = [60, 61, 62, 63, 64, 65, 51, 52, 53]
dependecies[77] = [60, 61, 62, 63, 64, 65, 51, 52, 53]

dependecies[78] = [66, 67]
dependecies[79] = [68, 69]

dependecies[80] = [70, 66]
dependecies[81] = [71, 67]
dependecies[82] = [72, 73]
dependecies[83] = [74, 68]
dependecies[84] = [75, 69]
dependecies[85] = [76, 77]

dependecies[86] = [80, 81]
dependecies[87] = [82, 78]
dependecies[88] = [83, 84]
dependecies[89] = [85, 79]

dependecies[90] = [86, 87]
dependecies[91] = [88, 89]

dependecies[92] = [78, 91]
dependecies[93] = [79, 90]

dependecies[94] = [92, 93]
dependecies[95] = [92, 93]

dayless = [0 for ii in range(len(dependecies))]
dayless[94] = -1

dep['cup-ourb'] = dependecies
deyl['cup-ourb'] = dayless

dependecies = {}
for ii in range(24):
    dependecies[ii] = []

for ii in range(12):
    dependecies[24+ii] = [ii*2, ii*2+1]
    dependecies[36+ii] = [ii*2, ii*2+1]

for ii in range(6):
    dependecies[48+ii] = [24+ii*2, 24+ii*2+1]
for ii in range(12):
    dependecies[54+ii] = [24+ii, 36+ii]

dependecies[66] = [48, 49]
dependecies[67] = [50, 60, 61, 62, 63, 64, 65, 51, 52, 53]
dependecies[68] = [51, 52]
dependecies[69] = [53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

dependecies[70] = [54, 55, 56, 57, 58, 59, 60, 61, 62]
dependecies[71] = [54, 55, 56, 57, 58, 59, 60, 61, 62]
dependecies[72] = [54, 55, 56, 57, 58, 59, 60, 61, 62]
dependecies[73] = [54, 55, 56, 57, 58, 59, 60, 61, 62]
dependecies[74] = [60, 61, 62, 63, 64, 65, 51, 52, 53]
dependecies[75] = [60, 61, 62, 63, 64, 65, 51, 52, 53]
dependecies[76] = [60, 61, 62, 63, 64, 65, 51, 52, 53]
dependecies[77] = [60, 61, 62, 63, 64, 65, 51, 52, 53]

dependecies[78] = [66, 67]
dependecies[79] = [68, 69]

dependecies[80] = [70, 68]
dependecies[81] = [71, 69]
dependecies[82] = [72, 73]
dependecies[83] = [74, 66]
dependecies[84] = [75, 67]
dependecies[85] = [76, 77]

dependecies[86] = [80, 81]
dependecies[87] = [82, 79]
dependecies[88] = [83, 84]
dependecies[89] = [85, 78]

dependecies[90] = [86, 87]
dependecies[91] = [88, 89]

dependecies[92] = [78, 90]
dependecies[93] = [79, 91]

dependecies[94] = [92, 93]
dependecies[95] = [92, 93]

dayless = [0 for ii in range(len(dependecies))]
for ii in range(36, 36+12):
    dayless[ii] = -1
for ii in range(54, 54+12):
    dayless[ii] = -1
for ii in range(70, 70+8):
    dayless[ii] = -1
for ii in range(80, 80+12):
    dayless[ii] = -1
dayless[94] = -1

# dep['cup-our2'] = dependecies
# deyl['cup-our2'] = dayless

# cup-3
dependecies = {}
for ii in range(16):
    dependecies[ii] = []

for ii in range(16):
    dependecies[16+ii] = [ii]
    dependecies[32+ii] = [16+ii]


for ii in range(16):
    dependecies[48+ii*2] = [32+ii*2, 32+ii*2+1]
    dependecies[48+ii*2+1] = [32+ii*2, 32+ii*2+1]

dependecies[64] = [48, 50]
dependecies[64+1] = [49, 51]
dependecies[64+2] = [52, 54]
dependecies[64+3] = [53, 55]
dependecies[64+4] = [56, 58]
dependecies[64+5] = [57, 59]
dependecies[64+6] = [60, 62]
dependecies[64+7] = [61, 63]

dependecies[72] = [64, 66]
dependecies[72+1] = [65, 67]
dependecies[72+2] = [68, 70]
dependecies[72+3] = [69, 71]

dependecies[76] = [72, 74]
dependecies[77] = [73, 75]

dependecies[78] = [76, 77]
dependecies[79] = [76, 77]

dayless = [0 for ii in range(len(dependecies))]
dayless[-2] = -1

dep['cup-3'] = dependecies
deyl['cup-3'] = dayless

# cup-4
dependecies = {}
for ii in range(24):
    dependecies[ii] = []

for ii in range(12):
    dependecies[24+ii*2] = [ii*2, ii*2+1]
    dependecies[24+ii*2+1] = [ii*2, ii*2+1]
    dependecies[48+ii*2] = [24+2*ii, 24+2*ii+1]
    dependecies[48+ii*2+1] = [24+2*ii, 24+2*ii+1]

# 72

dependecies[72] = list(np.arange(48, 48+12))
dependecies[73] = [48+2, 48+3, 48+10, 48+11]
dependecies[74] = [48+10, 48+11, 48+8, 48+9]
dependecies[75] = list(np.arange(48, 48+12))
dependecies[76] = [48+6, 48+7, 48+8, 48+9]
dependecies[77] = list(np.arange(48, 48+12))
dependecies[78] = list(np.arange(48, 48+12))
dependecies[79] = [48+0, 48+1, 48+4, 48+5]
dependecies[80] = list(np.arange(48+12, 48+12+12))
dependecies[81] = [48+12+2, 48+12+3, 48+12+10, 48+12+11]
dependecies[82] = [48+12+10, 48+12+11, 48+12+8, 48+12+9]
dependecies[83] = list(np.arange(48+12, 48+12+12))
dependecies[84] = [48+12+6, 48+12+7, 48+12+8, 48+12+9]
dependecies[85] = list(np.arange(48+12, 48+12+12))
dependecies[86] = list(np.arange(48+12, 48+12+12))
dependecies[87] = [48+12+0, 48+12+1, 48+12+4, 48+12+5]

for ii in range(8):
    dependecies[88+ii] = [72+ii*2, 72+ii*2+1]

for ii in range(4):
    dependecies[96+ii] = [88+ii*2, 88+ii*2+1]

dependecies[100] = [96, 97]
dependecies[101] = [98, 99]

dependecies[102] = [100, 101]
dependecies[103] = [100, 101]

dep['cup-4'] = dependecies

dayless = [0 for ii in range(len(dependecies))]
dayless[-2] = -1
deyl['cup-4'] = dayless


def arruma_sequencia(seq, dependecies):

    novaseq = []
    aguarda = [kk for kk in seq]

    while (len(aguarda) > 0):
        novaaguarda = []
        for ii in aguarda:
            if ii not in novaseq:
                tuput = True
                for kk in dependecies[ii]:
                    if kk not in novaseq:
                        tuput = False
                if (tuput):
                    novaseq.append(ii)
                else:
                    novaaguarda.append(ii)
        aguarda = [kk for kk in novaaguarda]

    return novaseq


def make_a_schedulle(seq, dependencies, dayless=None, maxpordia=4, mindelay=4):
    que_dia = [-1 for ii in range(len(seq))]
    quais_no_dia = {}
    for ss in seq:
        minday = 0
        minday2 = 0
        if dayless == None:
            mindelayb = mindelay
        else:
            mindelayb = mindelay + dayless[ss]
        for dd in dependencies[ss]:
            if (que_dia[dd] == -1):
                minday = 99
                minday2 = minday + mindelayb
            if (que_dia[dd] >= minday):
                minday = que_dia[dd]
                minday2 = minday + mindelayb
        minday = minday2
        achei = False
        while (achei == False):
            if (minday not in quais_no_dia.keys()):
                quais_no_dia[minday] = [ss]
                que_dia[ss] = minday
                achei = True
            elif (len(quais_no_dia[minday]) >= maxpordia):
                minday = minday + 1
            else:
                quais_no_dia[minday].append(ss)
                que_dia[ss] = minday
                achei = True

    return que_dia, quais_no_dia


def cross_over(seq1, seq2, dependencies):
    crosspoint = np.random.randint(len(seq1))
    newseq = [ss for ss in seq1[:crosspoint]]
    for ss in seq2:
        if ss not in newseq:
            newseq.append(ss)
    newseq = arruma_sequencia(newseq, dependencies)
    return newseq


def mutation(seq1, dependencies):
    crosspoint1 = np.random.randint(len(seq1))
    crosspoint2 = np.random.randint(len(seq1))
    ttt = seq1[crosspoint1]
    seq1[crosspoint1] = seq1[crosspoint2]
    seq1[crosspoint2] = ttt
    newseq = arruma_sequencia(seq1, dependencies)
    return newseq


def busca_ga(dependencies, pop_size=200, generations=50, cross=50, muta=50, maxpordia=4, mindelay=4, dayless=None):

    seqmat = [arruma_sequencia(np.random.permutation(
        len(dependencies.keys())), dependencies) for ii in range(pop_size)]
    fitness = [0 for ii in range(len(seqmat))]
    for ii in range(len(seqmat)):
        aaa, bbb = make_a_schedulle(
            seqmat[ii], dependencies, dayless=dayless, maxpordia=maxpordia, mindelay=mindelay)
        fitness[ii] = np.max(aaa)

    vvv = [np.min(fitness)]

    for gg in range(generations):
        salvado = pop_size-cross-muta
        quais = np.argsort(fitness)
        seqmat = list(np.array(seqmat)[quais][:salvado])
        # fitness = list(np.array(fitness)[quais][:salvado])
        for ii in range(muta):
            seqmat.append(
                mutation(seqmat[np.random.randint(salvado)], dependencies))
        for ii in range(cross):
            seqmat.append(cross_over(seqmat[np.random.randint(
                salvado)], seqmat[np.random.randint(salvado)], dependencies))

        # for ii in range(salvado,pop_size):
        #  aaa,bbb = make_a_schedulle(seqmat[ii],dependencies,maxpordia = maxpordia,mindelay = mindelay)
        #  fitness.append(np.max(aaa))
        fitness = [0 for ii in range(len(seqmat))]
        for ii in range(len(seqmat)):
            aaa, bbb = make_a_schedulle(
                seqmat[ii], dependencies, dayless=dayless, maxpordia=maxpordia, mindelay=mindelay)
            fitness[ii] = np.max(aaa)

        vvv.append(np.min(fitness))

    quais = np.argsort(fitness)
    seqmat = list(np.array(seqmat)[quais])
    fitness = list(np.array(fitness)[quais])

    return vvv, seqmat[0]


fff = 'cup-ourb'
dependecies = dep[fff]
# rrr = np.random.permutation(len(dependecies.keys()))
# rrr = arruma_sequencia(rrr,dependecies)
que_dia, quais_no_dia = make_a_schedulle(np.arange(
    len(dependecies.keys())), dependecies, dayless=deyl[fff], maxpordia=6, mindelay=4)
print(fff)
print(np.max(que_dia))
opo = np.sort(list(quais_no_dia.keys()))
for vvv in opo:
    print(str(vvv) + ':' + str(quais_no_dia[vvv]))

np.array(list(quais_no_dia.keys()))

fff = 'cup-our'
dependencies = dep[fff]
pop_size = 10
seqmat = [arruma_sequencia(np.random.permutation(
    len(dependencies.keys())), dependencies) for ii in range(pop_size)]
fitness = [0 for ii in range(len(seqmat))]
for ii in range(len(seqmat)):
    aaa, bbb = make_a_schedulle(
        seqmat[ii], dependencies, maxpordia=4, mindelay=4)
    fitness[ii] = np.max(aaa)

for pp in dep.keys():
    dependecies = dep[pp]
    ddd = deyl[pp]
    ddd = None
    que_dia, quais_no_dia = make_a_schedulle(np.arange(
        len(dependecies.keys())), dependecies, dayless=ddd, maxpordia=24, mindelay=4)
    print(pp)
    print(np.max(que_dia)+1)

lenght_games = {'range': np.arange(2, 26)}
for pp in dep.keys():
    lenght_games[pp] = [0 for ii in lenght_games['range']]

for xxi, games_per_day in enumerate(lenght_games['range']):
    for pp in dep.keys():
        dependecies = dep[pp]
        ddd = deyl[pp]
        # ddd = None
        que_dia, quais_no_dia = make_a_schedulle(np.arange(len(dependecies.keys(
        ))), dependecies, dayless=ddd, maxpordia=games_per_day, mindelay=4)
        # print(pp)
        # print(np.max(que_dia)+1)
        lenght_games[pp][xxi] = np.max(que_dia)+1

plt.figure(figsize=(10, 4))
# plt.axis('off')
plt.plot(lenght_games['range'], lenght_games['cup-ourb'],
         'r', label='Double elimination')
plt.plot(lenght_games['range'], lenght_games['cup-our'],
         'r--', label='Double elimination*')
plt.plot(lenght_games['range'], lenght_games['cup-3'], 'k', label='Group of 3')
plt.plot(lenght_games['range'], lenght_games['cup-4'],
         'k--', label='Group of 4')

plt.plot([1.5, 25.5], [32, 32], 'k--', alpha=0.2, linewidth=0.7)
plt.plot([1.5, 25.5], [39, 39], 'k--', alpha=0.2, linewidth=0.7)
plt.plot([1.5, 25.5], [46, 46], 'k--', alpha=0.2, linewidth=0.7)

for ii in range(28, 47):
    plt.plot([1.5, 25.5], [ii, ii], 'k--', alpha=0.1, linewidth=0.7)

for ii in range(2, 25):
    plt.plot([ii, ii], [28, 60], 'k--', alpha=0.1, linewidth=0.7)

plt.xlabel('Number of games per day')
plt.ylabel('Cup duration (days)')
plt.title('Cup duration vs the number of games per day (4 days delay)')
plt.xlim(1, 26)
plt.yticks([32, 39, 46])
plt.xticks([2, 3, 4, 5, 6, 8, 12, 16, 24])

for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)

# plt.tick_params(axis='y',which='minor',bottom=False)
plt.legend()
plt.show()

for pp in dep.keys():
    dependecies = dep[pp]
    ddd = deyl[pp]
    ddd = None
    que_dia, quais_no_dia = make_a_schedulle(np.arange(
        len(dependecies.keys())), dependecies, dayless=ddd, maxpordia=7, mindelay=4)
    print(pp)
    print(np.max(que_dia))

maxdia = 4
mindelay = 4
vovo = {}
vuvu = {}
for vvv in dep.keys():

    ddd = deyl[vvv]

    vvv2, xxx = busca_ga(dep[vvv], maxpordia=maxdia,
                         mindelay=mindelay, dayless=ddd, generations=100)
    print(vvv2)
    print(xxx)

    que_dia, quais_no_dia = make_a_schedulle(
        xxx, dep[vvv], dayless=ddd, maxpordia=maxdia, mindelay=mindelay)
    print(que_dia)
    print(max(que_dia))

    print(' ')

    vovo[vvv] = vvv2[-1]
    vuvu[vvv] = xxx

fff = 'cup-our'
dependecies = dep[fff]
ooo = vuvu[fff]
que_dia, quais_no_dia = make_a_schedulle(
    vuvu[fff], dependecies, maxpordia=4, mindelay=4)
quais_no_dia

fff = 'cup-our'
dependecies = dep[fff]
ooo = arruma_sequencia(vuvu[fff], dependecies)
que_dia, quais_no_dia = make_a_schedulle(
    ooo, dependecies, maxpordia=5, mindelay=4)
quais_no_dia

dep['cup-4'][100]

vvv = busca_ga(dep['cup-our'], maxpordia=5, mindelay=4)

vvv

vovo

"""# Draft"""

nn = 1000
eee = [0 for ii in range(nn)]
kk = 80
for oo in range(nn):
    zz = 0
    for ii in range(kk):
        aa = np.random.permutation(48)
        if (aa[0] < 8 and aa[1] < 8):
            zz = zz + 1
    eee[oo] = zz

np.random.permutation(5)

ccc = Cup48_3groups(teams_list)
ccc.run()
ttt = ccc.match_type_dist()
ttt

for ii in ccc.matches:
    print(ii.result.ranks)

ccc2 = Cup48(teams_list)
ccc2.run()
ttt = ccc2.match_type_dist()
ttt

print(len(ccc.matches))
print(len(ccc2.matches))

plt.hist(np.random.poisson(2.5, size=1000) -
         np.random.poisson(0.5, size=1000), bins=np.arange(-10, 11)-0.5)

ppp = 10000
zzz = []
ddd = []
eee = []
for ii in np.arange(0, 1.5, 0.1):
    ddf = np.random.poisson(1.5+ii, size=ppp) - \
        np.random.poisson(1.5-ii, size=ppp)
    zzz.append(np.sum(ddf > 0)/ppp)
    ddd.append(np.sum(ddf == 0)/ppp)
    eee.append(np.sum(ddf < 0)/ppp)

ooo = 50*np.arange(len(np.arange(0, 1.5, 0.1)))/len(np.arange(0, 1.5, 0.1))

plt.plot(ooo, 100*np.array(zzz))
plt.plot(ooo, 100*np.array(ddd))
plt.plot(ooo, 100*np.array(eee))

"""Cup48 - Standard"""

draw = Group(teams_list)
draw.draw()
groups = [draw]

ranks = {}
for ii in range(len(teams_list)):
    ranks[teams_list[ii]] = ii

gstart = [0]
matches = []
# Round 1
for ii in range(24):
    mmm = Match(gstart[0]+ii, ('p', 0, ii*2), ('p', 0, ii*2+1))
    mmm.setup(groups=groups, ranks=ranks)
    mmm.result.play()
    matches.append(mmm)
gstart.append(len(matches))

# Round 2
for ii in range(12):
    mmm = Match(gstart[1]+ii, ('w', ii*2), ('w', ii*2+1))
    mmm.setup(matches=matches, ranks=ranks)
    mmm.result.play()
    matches.append(mmm)
gstart.append(len(matches))

# losers Round 1 - wild card
for ii in range(12):
    mmm = Match(gstart[2]+ii, ('l', ii*2), ('l', ii*2+1))
    mmm.setup(matches=matches, ranks=ranks)
    mmm.result.play()
    matches.append(mmm)
gstart.append(len(matches))

# Round 3
for ii in range(6):
    mmm = Match(gstart[3]+ii, ('w', gstart[1]+ii*2), ('w', gstart[1]+ii*2+1))
    mmm.setup(matches=matches, ranks=ranks)  # pq aqui?
    mmm.result.play()
    matches.append(mmm)
gstart.append(len(matches))

# losers Round 2 - wild card
xii = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
for ii in range(12):
    mmm = Match(gstart[4]+ii, ('w', gstart[2]+ii), ('l', gstart[1]+xii[ii]))
    mmm.setup(matches=matches, ranks=ranks)
    mmm.result.play()
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
teams_group1.extend([matches[gstart[3]+ii].result.loser for ii in range(3)])
teams_group1.extend([matches[gstart[4]+ii].result.winner for ii in range(6)])

# make the group
matches_group2 = []
matches_group2.extend([gstart[0]+ii for ii in range(12, 24)])
matches_group2.extend([gstart[1]+ii for ii in range(6, 12)])
matches_group2.extend([gstart[2]+ii for ii in range(6, 12)])
matches_group2.extend([gstart[3]+ii for ii in range(3, 6)])
matches_group2.extend([gstart[4]+ii for ii in range(6, 12)])

teams_group2 = []
teams_group2.extend([matches[gstart[3]+ii].result.loser for ii in range(3, 6)])
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
matches.append(Match(67, ('w', 50), ('p', 2, 0)))
matches.append(Match(68, ('w', 51), ('w', 52)))
matches.append(Match(69, ('w', 53), ('p', 2, 0)))
gstart.append(len(matches))

# losers round 3 - wild card
matches.append(Match(70, ('p', 2, 1), ('p', 2, 8)))
matches.append(Match(71, ('p', 2, 2), ('p', 2, 7)))
matches.append(Match(72, ('p', 2, 3), ('p', 2, 6)))
matches.append(Match(73, ('p', 2, 4), ('p', 2, 5)))

matches.append(Match(74, ('p', 1, 1), ('p', 1, 8)))
matches.append(Match(75, ('p', 1, 2), ('p', 1, 7)))
matches.append(Match(76, ('p', 1, 3), ('p', 1, 6)))
matches.append(Match(77, ('p', 1, 4), ('p', 1, 5)))
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
matches.append(Match(92, ('w', 78), ('w', 90)))
matches.append(Match(93, ('w', 79), ('w', 91)))
gstart.append(len(matches))

# 3rd
matches.append(Match(94, ('l', 92), ('l', 93)))
gstart.append(len(matches))

# final
matches.append(Match(95, ('w', 92), ('w', 93)))
gstart.append(len(matches))

for mm in range(gstart[-1], 96):
    matches[mm].setup(matches=matches, groups=groups, ranks=ranks)
    matches[mm].result.play()

sss = Schedulle(matches)

orderr = list(np.arange(24))
orderr.extend([24, 25, 36, 37, 26, 27, 38, 39, 28, 29, 40, 41,
              30, 31, 42, 43, 32, 33, 44, 45, 34, 35, 46, 47])
orderr.extend(list(np.arange(48, 96)))
# orderr.extend([48,49,50,54,55,56,57,58,59,51,52,53,60,61,66,63,64,65])
# orderr.extend([74,75,76,77,70,71,72,73])
# orderr.extend([66,67,68,69,78,79])
# orderr.extend([80,81,82])
# orderr.extend([83,84,85])
# orderr.extend([88,89])
# orderr.extend([86,87])
# orderr.extend([90,91])
# orderr.extend([92,93,94,95])

allowmore = list(0*np.ones(gstart[1]-gstart[0]))
allowmore.extend(0*np.ones(gstart[2]-gstart[1]))
allowmore.extend(-1*np.ones(gstart[3]-gstart[2]))
allowmore.extend(0*np.ones(gstart[4]-gstart[3]))
allowmore.extend(-1*np.ones(gstart[5]-gstart[4]))
allowmore.extend(0*np.ones(gstart[6]-gstart[5]))
allowmore.extend(-1*np.ones(gstart[7]-gstart[6]))
allowmore.extend(0*np.ones(gstart[8]-gstart[7]))
allowmore.extend(-1*np.ones(gstart[9]-gstart[8]))
allowmore.extend(-1*np.ones(gstart[10]-gstart[9]))
allowmore.extend(-1*np.ones(gstart[11]-gstart[10]))
allowmore.extend(0*np.ones(gstart[12]-gstart[11]))
allowmore.extend(-1*np.ones(gstart[13]-gstart[12]))
allowmore.extend(0*np.ones(gstart[14]-gstart[13]))


sss.compute(groups=groups, mindelay=4, maxday=4,
            order=orderr, maxref=allowmore)

dependecies = {}
for ii in range(24):
    dependecies[ii] = []

for ii in range(12):
    dependecies[24+ii] = [ii*2, ii*2+1]
    dependecies[36+ii] = [ii*2, ii*2+1]

for ii in range(6):
    dependecies[48+ii] = [24+ii*2, 24+ii*2+1]
for ii in range(12):
    dependecies[54+ii] = [24+ii, 36+ii]

dependecies[66] = [48, 49]
dependecies[67] = [50, 60, 61, 62, 63, 64, 65, 51, 52, 53]
dependecies[68] = [51, 52]
dependecies[69] = [53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

dependecies[70] = [54, 55, 56, 57, 58, 59, 60, 61, 62]
dependecies[71] = [54, 55, 56, 57, 58, 59, 60, 61, 62]
dependecies[72] = [54, 55, 56, 57, 58, 59, 60, 61, 62]
dependecies[73] = [54, 55, 56, 57, 58, 59, 60, 61, 62]
dependecies[74] = [60, 61, 62, 63, 64, 65, 51, 52, 53]
dependecies[75] = [60, 61, 62, 63, 64, 65, 51, 52, 53]
dependecies[76] = [60, 61, 62, 63, 64, 65, 51, 52, 53]
dependecies[77] = [60, 61, 62, 63, 64, 65, 51, 52, 53]

dependecies[78] = [66, 67]
dependecies[79] = [68, 69]

dependecies[80] = [70, 68]
dependecies[81] = [71, 69]
dependecies[82] = [72, 73]
dependecies[83] = [74, 66]
dependecies[84] = [75, 67]
dependecies[85] = [76, 77]

dependecies[86] = [80, 81]
dependecies[87] = [82, 79]
dependecies[88] = [83, 84]
dependecies[89] = [85, 78]

dependecies[90] = [86, 87]
dependecies[91] = [88, 89]

dependecies[92] = [78, 90]
dependecies[93] = [79, 91]

dependecies[94] = [92, 93]
dependecies[95] = [92, 93]

sss.days

len(allowmore)

"""Cup48 - Groups3"""

draw = Group(teams_list)
draw.draw3()
groups = [draw]

ranks = {}
for ii in range(len(teams_list)):
    ranks[teams_list[ii]] = ii

gstart = [0]
matches = []
# Round 1
for ii in range(16):
    mmm = Match(gstart[0]+ii, ('p', 0, ii*3), ('p', 0, ii*3+1))
    mmm.setup(groups=groups, ranks=ranks)
    mmm.result.play()
    matches.append(mmm)
gstart.append(len(matches))

# Round 1
for ii in range(16):
    mmm = Match(gstart[1]+ii, ('p', 0, ii*3+2), ('p', 0, ii*3))
    mmm.setup(groups=groups, ranks=ranks)
    mmm.result.play()
    matches.append(mmm)
gstart.append(len(matches))

# Round 1
for ii in range(16):
    mmm = Match(gstart[2]+ii, ('p', 0, ii*3+1), ('p', 0, ii*3+2))
    mmm.setup(groups=groups, ranks=ranks)
    mmm.result.play()
    matches.append(mmm)
gstart.append(len(matches))

# make the group
for ii in range(16):
    matches_group = [ii, ii+16, ii+32]
    teams_group = [teams_list[kk] for kk in [ii*3, ii*3+1, ii*3+2]]
    ggg = Group(teams_group, [matches[ii] for ii in matches_group])
    ggg.compute()
    groups.append(ggg)


# Round of 32-
matches.append(Match(48, ('p', 1, 1), ('p', 2, 2)))
matches.append(Match(49, ('p', 2, 1), ('p', 1, 2)))
matches.append(Match(50, ('p', 3, 1), ('p', 4, 2)))
matches.append(Match(51, ('p', 4, 1), ('p', 3, 2)))
matches.append(Match(52, ('p', 5, 1), ('p', 6, 2)))
matches.append(Match(53, ('p', 6, 1), ('p', 5, 2)))
matches.append(Match(54, ('p', 7, 1), ('p', 8, 2)))
matches.append(Match(55, ('p', 8, 1), ('p', 7, 2)))
matches.append(Match(56, ('p', 9, 1), ('p', 10, 2)))
matches.append(Match(57, ('p', 10, 1), ('p', 9, 2)))
matches.append(Match(58, ('p', 11, 1), ('p', 12, 2)))
matches.append(Match(59, ('p', 12, 1), ('p', 11, 2)))
matches.append(Match(60, ('p', 13, 1), ('p', 14, 2)))
matches.append(Match(61, ('p', 14, 1), ('p', 13, 2)))
matches.append(Match(62, ('p', 15, 1), ('p', 16, 2)))
matches.append(Match(63, ('p', 16, 1), ('p', 15, 2)))
gstart.append(len(matches))

for ii in range(48, 64):
    matches[ii].setup(matches=matches, ranks=ranks, groups=groups)
    matches[ii].result.play()

# Round of 16
matches.append(Match(64, ('w', 48), ('w', 50)))
matches.append(Match(65, ('w', 49), ('w', 51)))
matches.append(Match(66, ('w', 52), ('w', 54)))
matches.append(Match(67, ('w', 53), ('w', 55)))
matches.append(Match(68, ('w', 56), ('w', 58)))
matches.append(Match(69, ('w', 57), ('w', 59)))
matches.append(Match(70, ('w', 60), ('w', 62)))
matches.append(Match(71, ('w', 61), ('w', 63)))
gstart.append(len(matches))

for ii in range(64, 72):
    matches[ii].setup(matches=matches, ranks=ranks)
    matches[ii].result.play()


# Round of 8
matches.append(Match(72, ('w', 64), ('w', 66)))
matches.append(Match(73, ('w', 65), ('w', 67)))
matches.append(Match(74, ('w', 68), ('w', 70)))
matches.append(Match(75, ('w', 69), ('w', 71)))
gstart.append(len(matches))

for ii in range(72, 76):
    matches[ii].setup(matches=matches, ranks=ranks)
    matches[ii].result.play()

# Semi
matches.append(Match(76, ('w', 72), ('w', 74)))
matches.append(Match(77, ('w', 73), ('w', 75)))
gstart.append(len(matches))

for ii in range(76, 78):
    matches[ii].setup(matches=matches, ranks=ranks)
    matches[ii].result.play()

# 3rd
matches.append(Match(78, ('l', 76), ('l', 77)))
gstart.append(len(matches))

# final
matches.append(Match(79, ('w', 76), ('w', 77)))
gstart.append(len(matches))

for ii in range(78, 80):
    matches[ii].setup(matches=matches, ranks=ranks)
    matches[ii].result.play()

sss = Schedulle(matches)
sss.compute(groups=groups, mindelay=4, maxday=4)
sss.matchday

groups[matches[ii].fromHome[1]].position[matches[ii].fromHome[2]]

len(gstart)

teams_list = ['Brazil',
              'Belgium',
              'Argentina',
              'France',
              'England',
              'Italy',
              'Spain',
              'Netherlands',
              'Portugal',
              'Denmark',
              'Germany',
              'Croatia',
              'Mexico',
              'Uruguay',
              'Switzerland',
              'USA',
              'Colombia',
              'Senegal',
              'Wales',
              'IR Iran',
              'Serbia',
              'Morocco',
              'Peru',
              'Japan',
              'Sweden',
              'Poland',
              'Ukraine',
              'Korea Republic',
              'Chile',
              'Tunisia',
              'Costa Rica',
              'Nigeria',
              'Russia',
              'Austria',
              'Czech Republic',
              'Hungary',
              'Algeria',
              'Australia',
              'Egypt',
              'Scotland',
              'Canada',
              'Norway',
              'Cameroon',
              'Ecuador',
              'Türkiye',
              'Mali',
              'Paraguay',
              'Côte dIvoire']

draw = Group(teams_list)
draw.draw()

draw.position

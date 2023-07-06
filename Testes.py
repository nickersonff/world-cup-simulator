from FormatReader import FormatReader
from WorldCup import WorldCup
from Params import Params
import itertools as iter
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

Params.load_elo_rating(format="Cup48_4groups")
cup = WorldCup(teams_list=Params.teams_list, ranks=Params.ranklist)
cup.run(format="Cup48_4groups")
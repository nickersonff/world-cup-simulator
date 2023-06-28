from FormatReader import FormatReader
from WorldCup import WorldCup
from Params import Params
import itertools as iter

Params.load_elo_rating(format="Cup48_3groups")
print(Params.ranklist)
cup = WorldCup(teams_list=Params.teams_list, ranks=Params.ranklist)
cup.run(format="Cup48_4groups")

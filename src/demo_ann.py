import random, numpy, math
import src.tsp.annealing as tsp
from src.plot.utils import PlotData, PlotUtils
from src.tsp.cooling import CoolingType


N = 10

cities = [(x, random.sample(range(100), 2)) for x in range(N)]

matrix = numpy.zeros((N, N))
pairs = [(a, b) for a in cities for b in cities]
for idx, pair in enumerate(pairs):
    a, b = pair
    distance = math.hypot(a[1][0] - b[1][0], a[1][1] - b[1][1])
    matrix[a[0], b[0]] = distance
    matrix[b[0], a[0]] = distance
numpy.fill_diagonal(matrix, matrix.max() + 1)

annWthReduction = tsp.Annealing(CoolingType.CONSTANT, matrix=matrix, reduction=True)
annWoReduction = tsp.Annealing(CoolingType.CONSTANT, matrix=matrix, reduction=False)

for i in range(10):
    annWthReduction.anneal()
    annWoReduction.anneal()

tourWthRed = annWthReduction.route
tourWoRed = annWoReduction.route

datas = [PlotData([cities[tourWthRed[i % N]][1][0] for i in range(N + 1)],
                  [cities[tourWthRed[i % N]][1][1] for i in range(N + 1)],
                  color="firebrick", label="With reduction", plotvalue=annWthReduction.value),
         PlotData([cities[tourWoRed[i % N]][1][0] for i in range(N + 1)],
                  [cities[tourWoRed[i % N]][1][1] for i in range(N + 1)],
                  color="purple", label="Without reduction", plotvalue=annWoReduction.value)]
PlotUtils.build_plot(datas)

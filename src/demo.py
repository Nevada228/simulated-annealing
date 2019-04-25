import random, numpy, math
import src.tsp.annealing as tsp
from src.plot.utils import PlotData, PlotUtils
from src.tsp.cooling import CoolingType
from src.tsp.greedy import Greedy
from src.tsp.exhaustive import Exhaustive

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

annealing = tsp.Annealing(CoolingType.CONSTANT, matrix=matrix)
exhaustive = Exhaustive(matrix)
greedy = Greedy(matrix)

greedyTour, greedyVal = greedy.find_best_route()
exhTour, exhVal = exhaustive.find_best_route()
annTour, annVal = annealing.find_best_route()

datas = [PlotData([cities[annTour[i % N]][1][0] for i in range(N + 1)],
                  [cities[annTour[i % N]][1][1] for i in range(N + 1)],
                  color="blue", label="Annealing", plotvalue=annVal),
         PlotData([cities[greedyTour[i % N]][1][0] for i in range(N + 1)],
                  [cities[greedyTour[i % N]][1][1] for i in range(N + 1)],
                  color="green", label="Greedy", plotvalue=greedyVal),
         PlotData([cities[exhTour[i % N]][1][0] for i in range(N + 1)],
                  [cities[exhTour[i % N]][1][1] for i in range(N + 1)],
                  color="red", label="Exaustive", plotvalue=exhVal)]
PlotUtils.build_plot(datas)

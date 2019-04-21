import random, numpy, math, matplotlib.pyplot as plt
import src.tsp.annealing as tsp
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

test = tsp.Annealing(CoolingType.CONSTANT, matrix=matrix)

for i in range(10):
    test.anneal()

tour = test.route

plt.plot([cities[tour[i % N]][1][0] for i in range(N + 1)],
         [cities[tour[i % N]][1][1] for i in range(N + 1)], 'xb-')
plt.show()

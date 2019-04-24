from numpy.core.multiarray import ndarray
import itertools
from src.tsp.tsp import Tsp


class Exhaustive(Tsp):
    def __init__(self, matrix: ndarray):
        super().__init__()
        self.matrix = matrix

    def find_best_route(self) -> (ndarray, int):

        n = self.matrix.shape[0]
        self.route = [i for i in range(n)]
        best_value = self._evaluate(self.route)

        iterator = itertools.permutations(range(0, n))
        for x in iterator:
            route = list(x)
            value = self._evaluate(route)
            if value < best_value:
                self.route = route
                best_value = value
        return self.route, best_value
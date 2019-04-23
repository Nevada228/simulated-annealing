from numpy.core.multiarray import ndarray

from src.tsp.tsp import Tsp


class Greedy(Tsp):

    def __init__(self, matrix: ndarray):
        super().__init__()
        self.matrix = matrix

    def find_best_route(self) -> (ndarray, int):
        route = [0]
        value = 0
        current = 0

        for _ in self.matrix:
            idx = Greedy.get_min(self.matrix[current], route)
            if idx is None:
                value += self.matrix[current, 0]
                break
            route.append(idx)
            value += self.matrix[current, idx]
            current = idx

        return route, value

    @staticmethod
    def get_min(arr, exclude: list):
        min_idx = None

        for i in arr.argsort():
            if i not in exclude:
                min_idx = i
                break
        return min_idx







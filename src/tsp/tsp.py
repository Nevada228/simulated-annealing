import abc
from numpy import ndarray


class Tsp(abc.ABC):

    def __init__(self):
        self.matrix = None
        self.route = None

    @abc.abstractmethod
    def find_best_route(self) -> (ndarray, int):
        pass

    def _evaluate(self, route=None) -> int:
        if route is None:
            route = self.route
        prev = None
        first = None
        value = 0
        for to in route:
            if prev is None:
                first = to
                prev = to
                continue
            value += self.matrix[prev, to]
            prev = to
        value += self.matrix[prev, first]
        return value

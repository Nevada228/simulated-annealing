import abc
from numpy import ndarray


class Tsp(abc.ABC):

    def __init__(self):
        self.matrix = None

    @abc.abstractmethod
    def find_best_route(self) -> (ndarray, int):
        pass

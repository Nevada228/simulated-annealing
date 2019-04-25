import abc
import src.tsp.rand as rand
import numpy as np
from src.tsp.cooling import Cooling, CoolingType
from src.tsp.tsp import Tsp


class AbstractAnnealing(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def anneal(self):
        pass

    @abc.abstractmethod
    def _mutate(self):
        pass

    @abc.abstractmethod
    def _cooling(self, cooling_method):
        pass


class Annealing(AbstractAnnealing, Tsp):
    MAX_DIFFUSION = 0.4
    MIN_TEMPERATURE = 0.5
    MAX_TEMPERATURE = 1000
    DEFAULT_MATRIX_SIZE = 100
    DEFAULT_MATRIX_MAX_VALUE = 100
    REPEAT = 10

    def __init__(self, cooling_type: CoolingType, matrix=None, reduction=True):
        AbstractAnnealing.__init__(self)
        Tsp.__init__(self)

        self.reduction = reduction
        self.temperature = Annealing.MAX_TEMPERATURE
        cooling = Cooling(Annealing.MAX_TEMPERATURE, 1.01, cooling_type)
        self.cooling_method = cooling.get_selected()

        self.matrix = matrix

        if self.matrix is None:
            self.matrix = Annealing.generate_matrix()
        self.route = np.array([i for i in range(self.matrix.shape[0])])

        self.value = self._evaluate()

    def find_best_route(self) -> (np.ndarray, int):
        for i in range(Annealing.REPEAT):
            self.anneal()
        return self.route, self.value

    def anneal(self):
        self.temperature = Annealing.MAX_TEMPERATURE
        iteration = 1
        best_value, best_route = self.value, self.route
        while self.temperature > Annealing.MIN_TEMPERATURE:
            guess = self._mutate()
            score = self._evaluate(guess)
            if score <= best_value:
                self.value, self.route = score, guess
                best_value, best_route = score, guess
            elif self._should_change(score - best_value):
                self.value, self.route = score, guess
            self.temperature = self._cooling(iteration)
            iteration += 1
        # print(iteration)
        self.value, self.route = best_value, best_route

    def _should_change(self, delta) -> bool:
        probability = np.exp(-delta / self.temperature)
        return True if np.random.rand() <= probability else False

    def _mutate(self):
        n = self._diffusion_rate() if self.reduction else self._fixed_size()
        new_route = np.copy(self.route)
        Annealing._swap_n(new_route, n)
        return new_route

    def _diffusion_rate(self) -> int:
        delta = Annealing.MAX_TEMPERATURE - Annealing.MIN_TEMPERATURE
        k = ((self.temperature / delta) * Annealing.MAX_DIFFUSION * self.route.size) / 2 + 1
        return int(k)

    def _fixed_size(self) -> int:
        return int((self.route.size * Annealing.MAX_DIFFUSION) / 2 + 1)

    def _cooling(self, i):
        return self.cooling_method(i, self.temperature)

    @staticmethod
    def _swap(route, pair: list):
        a, b = pair
        temp = route[a]
        route[a] = route[b]
        route[b] = temp

    @staticmethod
    def _swap_n(route, n=1):
        idxs = np.random.choice(route.size, (n, 2), replace=False)
        for pair in idxs:
            Annealing._swap(route, pair)

    # def _evaluate(self, route=None):
    #     if route is None:
    #         route = self.route
    #     it = np.nditer(route, flags=['f_index'])
    #     value = 0
    #     while not it.finished:
    #         value += self.matrix[it.index, it[0]]
    #         it.iternext()
    #     return value

    @staticmethod  # deprecated
    def isvalid(route) -> int:
        if route.size == 1:
            return True

        pairs = []
        for indexes, value in np.ndenumerate(route):
            pairs.append((indexes[0], value))
        pairs = dict(pairs)  # todo: сразу в dictionary (проверить на возможные ошибки)

        _in = []
        out = []

        cur = 0
        for count in range(len(pairs)):
            if cur != 0 and count == 0:
                return False
            _in.append(pairs[cur])
            out.append(cur)
            cur = pairs[cur]

        return len(_in) == len(set(_in)) and len(out) == len(set(out))

    @staticmethod
    def generate_matrix():
        matrix = rand.get_symmetric_matrix(Annealing.DEFAULT_MATRIX_SIZE, 1,
                                           Annealing.DEFAULT_MATRIX_MAX_VALUE)
        np.fill_diagonal(matrix, Annealing.DEFAULT_MATRIX_MAX_VALUE + 1)
        return matrix

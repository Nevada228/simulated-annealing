import unittest
import numpy as np
import src.tsp.annealing as anl
from src.tsp.cooling import CoolingType
from src.tsp.greedy import Greedy


class TestTsp(unittest.TestCase):
    MATRIX = [
        [10, 5, 7, 1, 3, 5, 8],
        [3, 10, 6, 7, 8, 1, 6],
        [2, 5, 10, 9, 3, 5, 7],
        [5, 6, 5, 10, 6, 5, 6],
        [6, 7, 2, 5, 10, 7, 9],
        [2, 6, 4, 2, 5, 10, 7],
        [3, 2, 5, 6, 8, 9, 10]
    ]

    def test_init_random(self):
        obj = anl.Annealing(CoolingType.CONSTANT)

    def test_init_by_matrix(self):
        obj = anl.Annealing(CoolingType.CONSTANT, matrix=np.array([[10, 3, 2], [5, 10, 1], [6, 4, 10]]))
        self.assertEqual(obj.value, 10)

    @unittest.skip("deprecated method")
    def test_validate_route(self):
        obj = anl.Annealing(CoolingType.CONSTANT, matrix=np.array([[10, 3, 2], [5, 10, 1], [6, 4, 10]]))
        self.assertTrue(anl.Annealing.isvalid(obj.route))
        self.assertTrue(anl.Annealing.isvalid(np.array([0])))  # один город
        self.assertTrue(anl.Annealing.isvalid(np.array([3, 2, 0, 1])))  # валидный маршрут
        self.assertTrue(anl.Annealing.isvalid(np.array([1, 2, 3, 0])))  # валидный маршрут
        self.assertFalse(anl.Annealing.isvalid(np.array([1, 1, 1, 1, 1])))  # все в один
        self.assertFalse(anl.Annealing.isvalid(np.array([1, 2, 3, 4, 2])))  # 2 захода в город
        self.assertFalse(anl.Annealing.isvalid(np.array([1, 5, 0, 4, 6, 1, 3])))  # подциклы
        self.assertFalse(anl.Annealing.isvalid(np.array([1, 5, 0, 4, 6, 1, 3])))

    def test_compare_solutions(self):

        exact_solution = 19
        ann = anl.Annealing(CoolingType.CONSTANT, matrix=(np.array(TestTsp.MATRIX)))
        ann.anneal()
        self.assertTrue(exact_solution <= ann.value)

    def test_greedy(self):
        greedy = Greedy(np.array(TestTsp.MATRIX))
        route, value = greedy.find_best_route()
        self.assertEqual(route, [0, 3, 2, 4, 1, 5, 6])
        self.assertEqual(value, 27)


if __name__ == '__main__':
    unittest.main()

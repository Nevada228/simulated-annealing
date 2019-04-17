import unittest
import numpy as np
import src.tsp.annealing as anl


class TestTsp(unittest.TestCase):

    def test_init_random(self):
        obj = anl.Tsp()

    def test_init_by_matrix(self):
        obj = anl.Tsp(matrix=np.array([[10, 3, 2], [5, 10, 1], [6, 4, 10]]))
        self.assertEqual(obj.value, 10)

    def test_validate_route(self):
        # deprecated
        obj = anl.Tsp(matrix=np.array([[10, 3, 2], [5, 10, 1], [6, 4, 10]]))
        self.assertTrue(anl.Tsp.isvalid(obj.route))
        self.assertTrue(anl.Tsp.isvalid(np.array([0])))  # один город
        self.assertTrue(anl.Tsp.isvalid(np.array([3, 2, 0, 1])))  # валидный маршрут
        self.assertTrue(anl.Tsp.isvalid(np.array([1, 2, 3, 0])))  # валидный маршрут
        self.assertFalse(anl.Tsp.isvalid(np.array([1, 1, 1, 1, 1])))  # все в один
        self.assertFalse(anl.Tsp.isvalid(np.array([1, 2, 3, 4, 2])))  # 2 захода в город
        self.assertFalse(anl.Tsp.isvalid(np.array([1, 5, 0, 4, 6, 1, 3])))  # подциклы
        self.assertFalse(anl.Tsp.isvalid(np.array([1, 5, 0, 4, 6, 1, 3])))

    def test_compare_solutions(self):
        matrix = [
            [10, 5, 7, 1, 3, 5, 8],
            [3, 10, 6, 7, 8, 1, 6],
            [2, 5, 10, 9, 3, 5, 7],
            [5, 6, 5, 10, 6, 5, 6],
            [6, 7, 2, 5, 10, 7, 9],
            [2, 6, 4, 2, 5, 10, 7],
            [3, 2, 5, 6, 8, 9, 10]
        ]
        exact_solution = 19
        ann = anl.Tsp(matrix=(np.array(matrix)))
        ann.anneal()
        self.assertTrue(exact_solution <= ann.value)


if __name__ == '__main__':
    unittest.main()

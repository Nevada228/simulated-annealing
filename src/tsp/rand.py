import numpy as np


def get_random_matrix(n: int = 5, m: int = 5, low=5, high=10):
    """
        Функция возвращающая матрицу из случайных элементов
    :param n: - количество столбцов
    :param m: - количество строк
    :param low: - нижняя граница случайного числа
    :param high: - верхняя граница случайного числа
    :return: матрица
    """
    return ((high - low) * np.random.sample((n, m)) + low).astype(int)


def get_random_vector(n: int = 5, low=0.0, high=1.0):
    """
        Функция возвращающая вектор из случайных элементов
    :param n: - размер вектора
    :param low:- нижняя граница случайного числа
    :param high:- верхняя граница случайного числа
    :return: вектор
    """
    return ((high - low) * np.random.rand(n) + low).astype(int)

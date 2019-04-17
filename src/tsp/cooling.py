from enum import Enum


class CoolingType(Enum):
    LINEAR = 1
    MAX_LINEAR = 2
    SUBTRACT = 3
    MAX_SUBTRACT = 4
    ALPHA = 5
    CONSTANT = 6


class Cooling:
    def __init__(self, tmax, alpha=1.001, ctype=CoolingType.LINEAR):
        self._selected = ctype
        self._tmax = tmax
        self._alpha = alpha

    def get_selected(self):
        if self._selected == CoolingType.MAX_LINEAR:
            return self.max_linear()
        elif self._selected == CoolingType.SUBTRACT:
            return self.subtract()
        elif self._selected == CoolingType.MAX_SUBTRACT:
            return self.max_subtract()
        elif self._selected == CoolingType.ALPHA:
            return self.alpha_descent()
        elif self._selected == CoolingType.CONSTANT:
            return self.constant()
        return self.linear()

    def linear(self):
        return lambda i, t: t / i

    def max_linear(self):
        return lambda i, t: self._tmax / i

    def subtract(self):
        return lambda i, t: t - i

    def max_subtract(self):
        return lambda i, t: self._tmax - i

    def alpha_descent(self):
        return lambda i, t: t / self._alpha ** i

    def constant(self): return lambda i, t: t - 1

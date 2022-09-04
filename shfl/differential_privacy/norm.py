import numpy as np
import abc
from multipledispatch import dispatch


class SensitivityNorm(abc.ABC):
    """
    This class defines the interface that must be implemented to compute
    the sensitivity norm between two values in a normed space.

    # Arguments:
        axis: direction. Options are axis=None that considers all elements
            and thus returns a scalar value for each array (default).
            Instead, axis=0 operates along vertical axis and thus returns a vector of
            size equal to the number of columns of each array
            (see [numpy.sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html))
    """

    def __init__(self, axis=None):
        self._axis = axis

    def compute(self, x_1, x_2):
        """
        The compute method receives the result of a query over private data and
        returns the norm of the difference between responses.
        # Arguments:
            x_1: array response from a concrete query over database 1
            x_2: array response from the same query over database 2
        """


class L1SensitivityNorm(SensitivityNorm):
    """
    Implements the L1 norm of the difference between x_1 and x_2
    """
    @dispatch((np.ScalarType, np.ndarray), (np.ScalarType, np.ndarray))
    def compute(self, x_1, x_2):
        """L1 norm of difference between arrays"""
        x = np.sum(np.abs(x_1 - x_2), axis=self._axis)
        return x

    @dispatch(list, list)
    def compute(self, x_1, x_2):
        """L1 norm of difference between (nested) lists of arrays"""
        x = [self.compute(xi_1, xi_2)
             for xi_1, xi_2 in zip(x_1, x_2)]
        return x


class L2SensitivityNorm(SensitivityNorm):
    """
    Implements the L2 norm of the difference between x_1 and x_2.
    """

    @dispatch((np.ScalarType, np.ndarray), (np.ScalarType, np.ndarray))
    def compute(self, x_1, x_2):
        """L2 norm of difference between arrays"""
        x = np.sqrt(np.sum((x_1 - x_2)**2, axis=self._axis))
        return x

    @dispatch(list, list)
    def compute(self, x_1, x_2):
        """L2 norm of difference between (nested) lists of arrays"""
        x = [self.compute(xi_1, xi_2)
             for xi_1, xi_2 in zip(x_1, x_2)]
        return x

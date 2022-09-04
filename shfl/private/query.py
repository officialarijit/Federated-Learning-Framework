import numpy as np
import abc


class Query(abc.ABC):
    """
    This class represents a query over private data. This interface exposes a method receiving
    data and must return a result based on this input.
    """

    @abc.abstractmethod
    def get(self, data):
        """
        Receives data and apply some function to answer it.

        # Arguments:
            data: Data to process

        # Returns:
            answer: Result of apply query over data
        """


class IdentityFunction(Query):
    """
    This function doesn't transform data. The answer is the data.
    """
    def get(self, data):
        return data


class Mean(Query):
    """
    Implements mean over data array.
    """
    def get(self, data):
        return np.mean(data)

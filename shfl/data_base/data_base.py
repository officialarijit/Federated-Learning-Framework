import abc
import numpy as np
import pandas as pd


def shuffle_rows(data, labels):
    """
    Shuffles rows in two data structures simultaneously.
    It supports either pd.DataFrame/pd.Series or numpy arrays.
    """
    randomize = np.arange(len(labels))
    np.random.shuffle(randomize)

    if isinstance(data, (pd.DataFrame, pd.Series)) and isinstance(labels, (pd.DataFrame, pd.Series)):
        data = data.iloc[randomize]
        labels = labels.iloc[randomize]

    elif isinstance(data, np.ndarray) and isinstance(labels, np.ndarray):
        data = data[randomize, ]
        labels = labels[randomize]

    else:
        raise TypeError("Data and labels must be either pd.DataFrame/pd.Series or numpy arrays.")

    return data, labels


def split_train_test(data, labels, dim):
    """
    Method that randomly chooses the train and test sets from data and labels.

    # Arguments:
        data: Data for extracting the validation data
        labels: Array with labels
        dim: Size for validation data

    # Returns:
        new_data: Data, labels, validation data and validation labels
    """

    data, labels = shuffle_rows(data, labels)

    test_data = data[0:dim]
    test_labels = labels[0:dim]

    rest_data = data[dim:]
    rest_labels = labels[dim:]

    return rest_data, rest_labels, test_data, test_labels


class DataBase(abc.ABC):
    """
    Abstract class for data base.

    Load method must be implemented in order to create a database able to \
    interact with the system, in concrete with data distribution methods \
    (see: [Data Distribution](../data_distribution)).

    Load method should save data in the protected Attributes:

    # Attributes:
        * **train_data, train_labels, test_data, test_labels**

    # Properties:
        train: Returns train data and labels
        test: Returns test data and labels
        data: Returns train data, train labels, validation data, validation labels, test data and test labels
    """

    def __init__(self):
        self._train_data = []
        self._test_data = []
        self._train_labels = []
        self._test_labels = []

    @property
    def train(self):
        return self._train_data, self._train_labels

    @property
    def test(self):
        return self._test_data, self._test_labels

    @property
    def data(self):
        return self._train_data, self._train_labels, self._test_data, self._test_labels

    @abc.abstractmethod
    def load_data(self):
        """
        Abstract method that loads the data
        """

    def shuffle(self):
        """
        Shuffles all data
        """
        self._train_data, self._train_labels = shuffle_rows(self._train_data, self._train_labels)
        self._test_data, self._test_labels = shuffle_rows(self._test_data, self._test_labels)


class LabeledDatabase(DataBase):
    """
    Class to create generic labeled database from data and labels vectors

    # Arguments
        data: Data features to load
        labels: Labels for this features
        train_percentage: float between 0 and 1 to indicate how much data is dedicated to train
    """
    def __init__(self, data, labels, train_percentage=0.8):
        super(DataBase, self).__init__()
        self._data = data
        self._labels = labels
        self._train_percentage = train_percentage

    def load_data(self):
        """
        Load data

        # Returns
            all_data : train data, train labels, test data and test labels
        """
        test_size = round(len(self._data) * (1 - self._train_percentage))
        self._train_data, self._train_labels, \
            self._test_data, self._test_labels = split_train_test(self._data, self._labels, test_size)

        self.shuffle()

        return self.data

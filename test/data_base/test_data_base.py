import numpy as np
import pandas as pd
import pytest

import shfl.data_base.data_base
from shfl.data_base.data_base import DataBase
from shfl.data_base.data_base import LabeledDatabase


class TestDataBase(DataBase):
    def __init__(self):
        super(TestDataBase, self).__init__()

    def load_data(self):
        self._train_data = np.random.rand(50).reshape([10, 5])
        self._test_data = np.random.rand(50).reshape([10, 5])
        self._train_labels = np.random.rand(10)
        self._test_labels = np.random.rand(10)


class TestDataBasePandas(DataBase):
    def __init__(self):
        super(TestDataBasePandas, self).__init__()

    def load_data(self):
        self._train_data = pd.DataFrame(np.random.rand(50).reshape([10, 5]))
        self._test_data = pd.DataFrame(np.random.rand(50).reshape([10, 5]))
        self._train_labels = pd.Series(np.random.rand(10))
        self._test_labels = pd.Series(np.random.rand(10))


def test_split_train_test():
    data = np.random.rand(50).reshape([10, -1])
    labels = np.random.rand(10)
    dim = 4

    rest_data, rest_labels, \
        validation_data, validation_labels = shfl.data_base.data_base.split_train_test(data, labels, dim)

    ndata = np.concatenate([rest_data, validation_data])
    nlabels = np.concatenate([rest_labels, validation_labels])

    data_ravel = np.sort(data.ravel())
    ndata_ravel = np.sort(ndata.ravel())

    assert np.array_equal(data_ravel, ndata_ravel)
    assert np.array_equal(np.sort(labels), np.sort(nlabels))
    assert rest_data.shape[0] == data.shape[0]-dim
    assert rest_labels.shape[0] == labels.shape[0]-dim
    assert validation_data.shape[0] == dim
    assert validation_labels.shape[0] == dim


def test_split_train_test_pandas():
    data = pd.DataFrame(np.random.rand(50).reshape([10, -1]))
    labels = pd.Series(np.random.rand(10))
    dim = 4

    rest_data, rest_labels, \
        validation_data, validation_labels = shfl.data_base.data_base.split_train_test(data, labels, dim)

    ndata = pd.concat([rest_data, validation_data])
    nlabels = pd.concat([rest_labels, validation_labels])

    data_ravel = data.sort_index()
    ndata_ravel = ndata.sort_index()

    assert np.array_equal(data_ravel, ndata_ravel)
    assert np.array_equal(np.sort(labels), np.sort(nlabels))
    assert rest_data.shape[0] == data.shape[0]-dim
    assert rest_labels.shape[0] == labels.shape[0]-dim
    assert validation_data.shape[0] == dim
    assert validation_labels.shape[0] == dim


def test_data_base_shuffle_elements():
    data = TestDataBase()
    data.load_data()

    train_data_b, train_labels_b, test_data_b, test_labels_b = data.data

    data.shuffle()

    train_data_a, train_labels_a, test_data_a, test_labels_a = data.data

    train_data_b = np.sort(train_data_b.ravel())
    train_data_a = np.sort(train_data_a.ravel())
    assert np.array_equal(train_data_b, train_data_a)

    test_data_b = np.sort(test_data_b.ravel())
    test_data_a = np.sort(test_data_a.ravel())
    assert np.array_equal(test_data_b, test_data_a)

    assert np.array_equal(np.sort(train_labels_b), np.sort(train_labels_a))
    assert np.array_equal(np.sort(test_labels_b), np.sort(test_labels_a))


def test_data_base_shuffle_elements_pandas():
    data = TestDataBasePandas()
    data.load_data()

    train_data_b, train_labels_b, test_data_b, test_labels_b = data.data

    data.shuffle()

    train_data_a, train_labels_a, test_data_a, test_labels_a = data.data

    train_data_b = train_data_b.sort_index()
    train_data_a = train_data_a.sort_index()
    assert np.array_equal(train_data_b, train_data_a)

    test_data_b = test_data_b.sort_index()
    test_data_a = test_data_a.sort_index()
    assert np.array_equal(test_data_b, test_data_a)

    assert np.array_equal(train_labels_b.sort_index(), train_labels_a.sort_index())
    assert np.array_equal(test_labels_b.sort_index(), test_labels_a.sort_index())


def test_data_base_shuffle_correct():
    data = TestDataBase()
    data.load_data()

    train_data_b, train_labels_b = data.train
    test_data_b, test_labels_b = data.test

    data.shuffle()

    train_data_a, train_labels_a = data.train
    test_data_a, test_labels_a = data.test

    assert (train_data_b == train_data_a).all() == False
    assert (test_data_b == test_data_a).all() == False


def test_data_base_shuffle_correct_pandas():
    data = TestDataBasePandas()
    data.load_data()

    train_data_b, train_labels_b = data.train
    test_data_b, test_labels_b = data.test

    data.shuffle()

    train_data_a, train_labels_a = data.train
    test_data_a, test_labels_a = data.test

    assert (train_data_b.to_numpy() == train_data_a.to_numpy()).all() == False
    assert (test_data_b.to_numpy() == test_data_a.to_numpy()).all() == False


def test_shuffle_wrong_call():
    data = TestDataBase()

    with pytest.raises(TypeError):
        data.shuffle()


def test_labeled_database():
    data = np.random.randint(low=0, high=100, size=100, dtype='l')
    labels = 10 + 2 * data + np.random.normal(loc=0.0, scale=10, size=len(data))
    database = LabeledDatabase(data, labels)
    loaded_data = database.load_data()

    assert loaded_data is not None
    assert len(loaded_data[1]) + len(loaded_data[3]) == len(data)


def test_split_wrong_type():
    data = np.random.rand(50).reshape([10, -1])
    labels = pd.Series(np.random.rand(10))
    dim = 4

    with pytest.raises(TypeError):
        shfl.data_base.data_base.split_train_test(data, labels, dim)

import numpy as np
import pandas as pd
import tensorflow as tf

from shfl.data_base.data_base import DataBase
from shfl.data_distribution.data_distribution_iid import IidDataDistribution


class TestDataBase(DataBase):
    def __init__(self):
        super(TestDataBase, self).__init__()

    def load_data(self):
        self._train_data = np.random.rand(200).reshape([40, 5])
        self._test_data = np.random.rand(200).reshape([40, 5])
        self._train_labels = tf.keras.utils.to_categorical(np.random.randint(0, 10, 40))
        self._test_labels = tf.keras.utils.to_categorical(np.random.randint(0, 10, 40))


class TestDataBasePandas(DataBase):
    def __init__(self):
        super(TestDataBasePandas, self).__init__()

    def load_data(self):
        self._train_data = pd.DataFrame(np.random.rand(200).reshape([40, 5]))
        self._test_data = pd.DataFrame(np.random.rand(200).reshape([40, 5]))
        self._train_labels = pd.DataFrame(tf.keras.utils.to_categorical(np.random.randint(0, 10, 40)))
        self._test_labels = pd.DataFrame(tf.keras.utils.to_categorical(np.random.randint(0, 10, 40)))


def test_make_data_federated():
    data = TestDataBase()
    data.load_data()
    data_distribution = IidDataDistribution(data)

    train_data, train_label = data_distribution._database.train

    num_nodes = 3
    percent = 60
    # weights = np.full(num_nodes, 1/num_nodes)
    weights = [0.5, 0.25, 0.25]
    federated_data, federated_label = data_distribution.make_data_federated(train_data,
                                                                            train_label,
                                                                            percent,
                                                                            num_nodes,
                                                                            weights)
    data_distribution.get_federated_data(3)

    all_data = np.concatenate(federated_data)
    all_label = np.concatenate(federated_label)

    idx = []
    for data in all_data:
        idx.append(np.where((data == train_data).all(axis=1))[0][0])

    for i, weight in enumerate(weights):
        assert federated_data[i].shape[0] == int(weight * int(percent * train_data.shape[0] / 100))

    assert all_data.shape[0] == int(percent * train_data.shape[0] / 100)
    assert num_nodes == len(federated_data) == len(federated_label)
    assert (np.sort(all_data.ravel()) == np.sort(train_data[idx,].ravel())).all()
    assert (np.sort(all_label, 0) == np.sort(train_label[idx], 0)).all()

    #test make federated data with replacement
    federated_data, federated_label = data_distribution.make_data_federated(train_data,
                                                                            train_label,
                                                                            percent,
                                                                            num_nodes,
                                                                            weights,
                                                                            sampling="with_replacement")
    all_data = np.concatenate(federated_data)
    all_label = np.concatenate(federated_label)

    idx = []
    for data in all_data:
        idx.append(np.where((data == train_data).all(axis=1))[0][0])

    for i, weight in enumerate(weights):
        assert federated_data[i].shape[0] == int(weight * int(percent * train_data.shape[0] / 100))

    assert all_data.shape[0] == int(percent * train_data.shape[0] / 100)
    assert num_nodes == len(federated_data) == len(federated_label)
    assert (np.sort(all_data.ravel()) == np.sort(train_data[idx,].ravel())).all()
    assert (np.sort(all_label, 0) == np.sort(train_label[idx], 0)).all()


def test_make_data_federated_pandas():
    data = TestDataBasePandas()
    data.load_data()
    data_distribution = IidDataDistribution(data)

    train_data, train_label = data_distribution._database.train

    num_nodes = 3
    percent = 60
    # weights = np.full(num_nodes, 1/num_nodes)
    weights = [0.5, 0.25, 0.25]
    federated_data, federated_label = data_distribution.make_data_federated(train_data,
                                                                            train_label,
                                                                            percent,
                                                                            num_nodes,
                                                                            weights)
    data_distribution.get_federated_data(3)

    all_data = pd.concat(federated_data)
    all_label = pd.concat(federated_label)

    for i, weight in enumerate(weights):
        assert federated_data[i].shape[0] == int(weight * int(percent * train_data.shape[0] / 100))

    assert all_data.shape[0] == int(percent * train_data.shape[0] / 100)
    assert num_nodes == len(federated_data) == len(federated_label)
    pd.testing.assert_frame_equal(all_data, train_data.iloc[all_data.index.values])
    pd.testing.assert_frame_equal(all_label, train_label.iloc[all_data.index.values])

    #test make federated data with replacement
    federated_data, federated_label = data_distribution.make_data_federated(train_data,
                                                                            train_label,
                                                                            percent,
                                                                            num_nodes,
                                                                            weights,
                                                                            sampling="with_replacement")
    all_data = pd.concat(federated_data)
    all_label = pd.concat(federated_label)

    for i, weight in enumerate(weights):
        assert federated_data[i].shape[0] == int(weight * int(percent * train_data.shape[0] / 100))

    assert all_data.shape[0] == int(percent * train_data.shape[0] / 100)
    assert num_nodes == len(federated_data) == len(federated_label)
    pd.testing.assert_frame_equal(all_data, train_data.iloc[all_data.index.values])
    pd.testing.assert_frame_equal(all_label, train_label.iloc[all_data.index.values])


def test_make_data_federated_wrong_weights():
    data = TestDataBase()
    data.load_data()
    data_distribution = IidDataDistribution(data)

    train_data, train_label = data_distribution._database.train

    num_nodes = 3
    percent = 60
    # weights = np.full(num_nodes, 1/num_nodes)
    weights = [0.5, 0.5, 0.5]
    federated_data, federated_label = data_distribution.make_data_federated(train_data,
                                                                            train_label,
                                                                            percent,
                                                                            num_nodes,
                                                                            weights)

    weights = np.array([float(i) / sum(weights) for i in weights])
    data_distribution.get_federated_data(3)

    all_data = np.concatenate(federated_data)
    all_label = np.concatenate(federated_label)

    idx = []
    for data in all_data:
        idx.append(np.where((data == train_data).all(axis=1))[0][0])

    for i, weight in enumerate(weights):
        assert federated_data[i].shape[0] == int(weight * int(percent * train_data.shape[0] / 100))

    assert all_data.shape[0] == int(percent * train_data.shape[0] / 100)
    assert num_nodes == len(federated_data) == len(federated_label)
    assert (np.sort(all_data.ravel()) == np.sort(train_data[idx, ].ravel())).all()
    assert (np.sort(all_label, 0) == np.sort(train_label[idx], 0)).all()

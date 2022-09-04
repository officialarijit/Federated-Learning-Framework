from shfl.data_base.emnist import Emnist


def test_emnist():
    data = Emnist()
    data.load_data()
    train_data, train_labels, test_data, test_labels = data.data

    assert train_data.size > 0
    assert test_data.size > 0
    assert train_data.shape[0] == len(train_labels)
    assert test_data.shape[0] == len(test_labels)

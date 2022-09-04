import numpy as np
import pytest

from shfl.model.mean_recommender import MeanRecommender


def test_mean_recommender():
    mean_recommender = MeanRecommender()

    assert mean_recommender._clientId is None


def test_train():
    data = np.array([[2, 3, 51],
                     [2, 34, 6],
                     [2, 33, 7],
                     [2, 13, 65],
                     [2, 3, 15]])
    labels = np.array([3, 2, 5, 6, 7])

    mean_recommender = MeanRecommender()
    mean_recommender.train(data, labels)

    assert mean_recommender._clientId == data[0, 0]
    assert mean_recommender._mu == np.mean(labels)


def test_train_wrong_data():
    data = np.array([[2, 3, 51],
                     [2, 34, 6],
                     [3, 33, 7],
                     [4, 13, 65],
                     [2, 3, 15]])
    labels = np.array([3, 2, 5, 6, 7])

    mean_recommender = MeanRecommender()
    with pytest.raises(AssertionError):
        mean_recommender.train(data, labels)


def test_train_wrong_data_labels():
    data = np.array([[2, 3, 51],
                     [2, 34, 6],
                     [2, 33, 7],
                     [2, 13, 65],
                     [2, 3, 15]])
    labels = np.array([3, 2, 5, 7])

    mean_recommender = MeanRecommender()
    with pytest.raises(AssertionError):
        mean_recommender.train(data, labels)


def test_predict():
    data = np.array([[2, 3, 51],
                     [2, 34, 6],
                     [2, 33, 7],
                     [2, 13, 65],
                     [2, 3, 15]])

    mean_recommender = MeanRecommender()
    mean_recommender._mu = 2.3
    predictions = mean_recommender.predict(data)

    assert np.array_equal(predictions, np.full(len(data), mean_recommender._mu))


def test_evaluate():
    data = np.array([[2, 3, 51],
                     [2, 34, 6],
                     [2, 33, 7],
                     [2, 13, 65],
                     [2, 3, 15]])
    labels = np.array([3, 2, 5, 6, 7])

    mean_recommender = MeanRecommender()
    mean_recommender._mu = 2.3
    rmse = mean_recommender.evaluate(data, labels)

    assert rmse == np.sqrt(np.mean((mean_recommender._mu - labels) ** 2))


def test_evaluate_no_data():
    data = np.empty((0, 3))
    labels = np.empty(0)

    mean_recommender = MeanRecommender()
    mean_recommender._mu = 2.3
    rmse = mean_recommender.evaluate(data, labels)

    assert rmse == 0


def test_performance():
    data = np.array([[2, 3, 51],
                     [2, 34, 6],
                     [2, 33, 7],
                     [2, 13, 65],
                     [2, 3, 15]])
    labels = np.array([3, 2, 5, 6, 7])

    mean_recommender = MeanRecommender()
    mean_recommender._mu = 2.3
    rmse = mean_recommender.performance(data, labels)

    assert rmse == np.sqrt(np.mean((mean_recommender._mu - labels) ** 2))


def test_performance_no_data():
    data = np.empty((0, 3))
    labels = np.empty(0)

    mean_recommender = MeanRecommender()
    mean_recommender._mu = 2.3
    rmse = mean_recommender.performance(data, labels)

    assert rmse == 0


def test_set_model_params():
    params = 3.4

    mean_recommender = MeanRecommender()
    mean_recommender.set_model_params(params)

    assert mean_recommender._mu == params


def test_get_model_params():
    mean_recommender = MeanRecommender()
    mean_recommender._mu = 3.5
    params = mean_recommender.get_model_params()

    assert mean_recommender._mu == params


def test_predict_wrong_data():
    data = np.array([[2, 3, 51],
                     [2, 34, 6],
                     [3, 33, 7],
                     [4, 13, 65],
                     [2, 3, 15]])

    mean_recommender = MeanRecommender()
    with pytest.raises(AssertionError):
        mean_recommender.predict(data)


def test_evaluate_wrong_data():
    data = np.array([[2, 3, 51],
                     [2, 34, 6],
                     [3, 33, 7],
                     [4, 13, 65],
                     [2, 3, 15]])
    labels = np.array([3, 2, 5, 6, 7])

    mean_recommender = MeanRecommender()
    with pytest.raises(AssertionError):
        mean_recommender.evaluate(data, labels)


def test_evaluate_wrong_data_labels():
    data = np.array([[2, 3, 51],
                     [2, 34, 6],
                     [2, 33, 7],
                     [2, 13, 65],
                     [2, 3, 15]])
    labels = np.array([3, 2, 5, 7])

    mean_recommender = MeanRecommender()
    with pytest.raises(AssertionError):
        mean_recommender.evaluate(data, labels)


def test_performance_wrong_data():
    data = np.array([[2, 3, 51],
                     [2, 34, 6],
                     [3, 33, 7],
                     [4, 13, 65],
                     [2, 3, 15]])
    labels = np.array([3, 2, 5, 6, 7])

    mean_recommender = MeanRecommender()
    with pytest.raises(AssertionError):
        mean_recommender.performance(data, labels)


def test_performance_wrong_data_labels():
    data = np.array([[2, 3, 51],
                     [2, 34, 6],
                     [2, 33, 7],
                     [2, 13, 65],
                     [2, 3, 15]])
    labels = np.array([3, 2, 5, 7])

    mean_recommender = MeanRecommender()
    with pytest.raises(AssertionError):
        mean_recommender.performance(data, labels)

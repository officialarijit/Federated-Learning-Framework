import pandas as pd
import numpy as np
import pytest

from shfl.model.content_based_recommender import ContentBasedRecommender


def test_content_based_recommender():
    df_items = pd.DataFrame({1: [23, 3, 2],
                             2: [7, 34, 3],
                             3: [8, 9, 43],
                             4: [3, 4, 2]})
    content_based_recommender = ContentBasedRecommender(df_items)
    df_items.index.name = "itemid"

    assert content_based_recommender._clientId is None
    assert content_based_recommender._mu is None
    assert content_based_recommender._profile is None
    pd.testing.assert_frame_equal(content_based_recommender._df_items, df_items)


def test_content_based_recommender_wrong_input():
    df_items = np.array([[23, 3, 2],
                         [7, 34, 3],
                         [8, 9, 43],
                         [3, 4, 2]])
    with pytest.raises(TypeError):
        ContentBasedRecommender(df_items)


def test_train():
    data = np.array([[2, 1],
                     [2, 1],
                     [2, 2],
                     [2, 2],
                     [2, 0]])
    labels = np.array([3, 2, 5, 6, 7])
    df_items = pd.DataFrame({1: [23, 3, 2],
                             2: [7, 34, 3],
                             3: [8, 9, 43],
                             4: [3, 4, 2]})

    content_based_recommender = ContentBasedRecommender(df_items)
    content_based_recommender.train(data, labels)
    df_data = pd.DataFrame(data, columns=["userid", "itemid"])
    df = df_data.join(df_items, on="itemid").drop(["userid", "itemid"], axis=1)

    assert content_based_recommender._clientId == data[0, 0]
    assert content_based_recommender._mu == np.mean(labels)
    np.testing.assert_equal(content_based_recommender._profile,
                            df.multiply(labels - np.mean(labels), axis=0).mean().values)


def test_train_wrong_number_of_columns():
    data = np.array([[2, 3, 51],
                     [2, 34, 6],
                     [2, 33, 7],
                     [2, 13, 65],
                     [2, 3, 15]])
    labels = np.array([3, 2, 5, 6, 7])
    df_items = pd.DataFrame({1: [23, 3, 2],
                             2: [7, 34, 3],
                             3: [8, 9, 43],
                             4: [3, 4, 2]})

    content_based_recommender = ContentBasedRecommender(df_items)
    with pytest.raises(AssertionError):
        content_based_recommender.train(data, labels)


def test_train_new_items():
    data = np.array([[2, 3],
                     [2, 34],
                     [2, 33],
                     [2, 13],
                     [2, 3]])
    labels = np.array([3, 2, 5, 6, 7])
    df_items = pd.DataFrame({1: [23, 3, 2],
                             2: [7, 34, 3],
                             3: [8, 9, 43],
                             4: [3, 4, 2]})

    content_based_recommender = ContentBasedRecommender(df_items)
    with pytest.raises(AssertionError):
        content_based_recommender.train(data, labels)


def test_predict():
    data = np.array([[2, 1],
                     [2, 1],
                     [2, 2],
                     [2, 2],
                     [2, 0]])
    df_items = pd.DataFrame({1: [23, 3, 2],
                             2: [7, 34, 3],
                             3: [8, 9, 43],
                             4: [3, 4, 2]})
    content_based_recommender = ContentBasedRecommender(df_items)
    content_based_recommender._mu = 1
    content_based_recommender._profile = np.array([1, 2, 3, 4])
    predictions = content_based_recommender.predict(data)

    df = content_based_recommender._join_dataframe_with_items_features(data)
    predictions_test = content_based_recommender._mu + df.values.dot(content_based_recommender._profile)

    np.testing.assert_equal(predictions, predictions_test)


def test_evaluate():
    data = np.array([[2, 1],
                     [2, 1],
                     [2, 2],
                     [2, 2],
                     [2, 0]])
    labels = np.array([3, 2, 5, 6, 7])
    df_items = pd.DataFrame({1: [23, 3, 2],
                             2: [7, 34, 3],
                             3: [8, 9, 43],
                             4: [3, 4, 2]})

    content_based_recommender = ContentBasedRecommender(df_items)
    content_based_recommender._mu = 1
    content_based_recommender._profile = np.array([4, 2, 3, 4])
    rmse = content_based_recommender.evaluate(data, labels)

    predictions = content_based_recommender.predict(data)
    rmse_test = np.sqrt(np.mean((predictions - labels) ** 2))

    assert rmse == rmse_test


def test_evaluate_no_data():
    data = np.empty((0, 2))
    labels = np.empty(0)

    df_items = pd.DataFrame({1: [23, 3, 2],
                             2: [7, 34, 3],
                             3: [8, 9, 43],
                             4: [3, 4, 2]})

    content_based_recommender = ContentBasedRecommender(df_items)
    content_based_recommender._mu = 2.3
    content_based_recommender._profile = np.array([4, 2, 3, 4])
    rmse = content_based_recommender.evaluate(data, labels)

    assert rmse == 0


def test_performance():
    data = np.array([[2, 1],
                     [2, 1],
                     [2, 2],
                     [2, 2],
                     [2, 0]])
    labels = np.array([3, 2, 5, 6, 7])
    df_items = pd.DataFrame({1: [23, 3, 2],
                             2: [7, 34, 3],
                             3: [8, 9, 43],
                             4: [3, 4, 2]})

    content_based_recommender = ContentBasedRecommender(df_items)
    content_based_recommender._mu = 1
    content_based_recommender._profile = np.array([4, 2, 3, 4])
    rmse = content_based_recommender.performance(data, labels)

    predictions = content_based_recommender.predict(data)
    rmse_test = np.sqrt(np.mean((predictions - labels) ** 2))

    assert rmse == rmse_test


def test_performance_no_data():
    data = np.empty((0, 2))
    labels = np.empty(0)

    df_items = pd.DataFrame({1: [23, 3, 2],
                             2: [7, 34, 3],
                             3: [8, 9, 43],
                             4: [3, 4, 2]})

    content_based_recommender = ContentBasedRecommender(df_items)
    content_based_recommender._mu = 2.3
    content_based_recommender._profile = np.array([4, 2, 3, 4])
    rmse = content_based_recommender.performance(data, labels)

    assert rmse == 0


def test_set_model_params():
    params = (3.4, np.array([2, 3, 5, 7]))
    df_items = pd.DataFrame({1: [23, 3, 2],
                             2: [7, 34, 3],
                             3: [8, 9, 43],
                             4: [3, 4, 2]})

    content_based_recommender = ContentBasedRecommender(df_items)
    content_based_recommender.set_model_params(params)

    assert content_based_recommender._mu == params[0]
    np.testing.assert_equal(content_based_recommender._profile, params[1])


def test_get_model_params():
    df_items = pd.DataFrame({1: [23, 3, 2],
                             2: [7, 34, 3],
                             3: [8, 9, 43],
                             4: [3, 4, 2]})

    content_based_recommender = ContentBasedRecommender(df_items)
    content_based_recommender._mu = 3.4
    content_based_recommender._profile = np.array([2, 3, 5, 7])
    params = content_based_recommender.get_model_params()

    assert content_based_recommender._mu == params[0]
    np.testing.assert_equal(content_based_recommender._profile, params[1])

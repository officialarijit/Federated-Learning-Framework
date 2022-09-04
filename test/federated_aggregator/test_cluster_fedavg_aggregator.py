import numpy as np
from unittest.mock import Mock, patch

from shfl.federated_aggregator.cluster_fedavg_aggregator import ClusterFedAvgAggregator


@patch('shfl.federated_aggregator.cluster_fedavg_aggregator.KMeans')
def test_aggregate_weights(mock_kmeans):
    cfa = ClusterFedAvgAggregator()

    model_aggregator = Mock()
    centers = np.random.rand(10)
    model_aggregator.cluster_centers_ = centers
    mock_kmeans.return_value = model_aggregator

    clients_params = np.random.rand(90).reshape((10, 3, 3))

    clients_params_array = np.concatenate((clients_params))
    n_clusters = clients_params[0].shape[0]

    res = cfa.aggregate_weights(clients_params)

    mock_kmeans.assert_called_once_with(n_clusters=n_clusters, init='k-means++')
    model_aggregator.fit.assert_called_once()
    np.testing.assert_array_equal(clients_params_array, model_aggregator.fit.call_args[0][0])
    assert isinstance(res, np.ndarray)
    assert np.array_equal(res, centers)


import numpy as np

from shfl.federated_aggregator.fedavg_aggregator import FedAvgAggregator


def test_aggregated_weights():
    num_clients = 10
    num_layers = 5
    tams = [[128, 64], [64, 64], [64, 64], [64, 32], [32, 10]]

    weights = []
    for i in range(num_clients):
        weights.append([np.random.rand(tams[j][0], tams[j][1]) for j in range(num_layers)])

    clients_params = np.array(weights)

    avgfa = FedAvgAggregator()
    aggregated_weights = avgfa.aggregate_weights(clients_params)

    own_agg = np.array([np.mean(clients_params[:, layer], axis=0) for layer in range(num_layers)])

    for i in range(num_layers):
        assert np.array_equal(own_agg[i], aggregated_weights[i])
    assert aggregated_weights.shape[0] == num_layers


def test_aggregated_weights_multidimensional_2D_array():
    num_clients = 10
    num_rows_params = 3
    num_cols_params = 9

    clients_params = []
    for i in range(num_clients):
        clients_params.append(np.random.rand(num_rows_params, num_cols_params))
    clients_params = np.array(clients_params)

    avgfa = FedAvgAggregator()
    aggregated_weights = avgfa.aggregate_weights(clients_params)

    own_agg = np.zeros((num_rows_params, num_cols_params))
    for i_client in range(num_clients):
        own_agg += clients_params[i_client]
    own_agg = own_agg / num_clients

    assert np.array_equal(own_agg, aggregated_weights)
    assert aggregated_weights.shape == own_agg.shape


def test_aggregated_weights_multidimensional_3D_array():
    num_clients = 10
    num_rows_params = 3
    num_cols_params = 9
    num_k_params = 5

    clients_params = []
    for i in range(num_clients):
        clients_params.append(np.random.rand(num_rows_params, num_cols_params, num_k_params))
    clients_params = np.array(clients_params)

    avgfa = FedAvgAggregator()
    aggregated_weights = avgfa.aggregate_weights(clients_params)

    own_agg = np.zeros((num_rows_params, num_cols_params, num_k_params))
    for i_client in range(num_clients):
        own_agg += clients_params[i_client]
    own_agg = own_agg / num_clients

    assert np.array_equal(own_agg, aggregated_weights)
    assert aggregated_weights.shape == own_agg.shape


def test_aggregated_weights_list_of_arrays():
    num_clients = 10

    clients_params = []
    for i_client in range(num_clients):
        clients_params.append([np.random.rand(30, 20),
                               np.random.rand(20, 30),
                               np.random.rand(50, 40)])

    avgfa = FedAvgAggregator()
    aggregated_weights = avgfa.aggregate_weights(clients_params)

    own_agg = [np.zeros((30, 20)),
               np.zeros((20, 30)),
               np.zeros((50, 40))]
    for i_client in range(num_clients):
        for i_params in range(len(clients_params[0])):
            own_agg[i_params] += clients_params[i_client][i_params]
    for i_params in range(len(clients_params[0])):
        own_agg[i_params] = own_agg[i_params] / num_clients

    for i_params in range(len(clients_params[0])):
        assert np.array_equal(own_agg[i_params], aggregated_weights[i_params])
        assert aggregated_weights[i_params].shape == own_agg[i_params].shape

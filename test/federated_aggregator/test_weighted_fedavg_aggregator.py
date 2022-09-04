import numpy as np

from shfl.federated_aggregator.weighted_fedavg_aggregator import WeightedFedAvgAggregator


def test_aggregated_weights():
    num_clients = 10
    num_layers = 5
    tams = [[128, 64], [64, 64], [64, 64], [64, 32], [32, 10]]

    weights = []
    for i in range(num_clients):
        weights.append([np.random.rand(tams[j][0], tams[j][1])
                        for j in range(num_layers)])

    clients_params = np.array(weights)

    percentage = np.random.dirichlet(np.ones(num_clients),size=1)[0]

    avgfa = WeightedFedAvgAggregator(percentage=percentage)
    aggregated_weights = avgfa.aggregate_weights(clients_params)

    own_ponderated_weights = np.array([percentage[client] * clients_params[client, :]
                                       for client in range(num_clients)])
    own_agg = np.array([np.sum(own_ponderated_weights[:, layer], axis=0)
                        for layer in range(num_layers)])

    for i in range(num_layers):
        assert np.array_equal(own_agg[i],aggregated_weights[i])
    assert aggregated_weights.shape[0] == num_layers


def test_weighted_aggregated_weights_list_of_arrays():
    num_clients = 10

    clients_params = []
    for i_client in range(num_clients):
        clients_params.append([np.random.rand(30, 20),
                               np.random.rand(20, 30),
                               np.random.rand(50, 40)])

    percentage = np.random.dirichlet(np.ones(num_clients), size=1)[0]
    avgfa = WeightedFedAvgAggregator(percentage=percentage)
    aggregated_weights = avgfa.aggregate_weights(clients_params)

    own_ponderated_weights = []
    for i_client in range(num_clients):
        own_ponderated_weights.append(
            [clients_params[i_client][i_params] * percentage[i_client]
             for i_params in range(len(clients_params[0]))])

    own_agg = [np.zeros((30, 20)),
               np.zeros((20, 30)),
               np.zeros((50, 40))]
    for i_client in range(num_clients):
        for i_params in range(len(clients_params[0])):
            own_agg[i_params] += own_ponderated_weights[i_client][i_params]

    for i_params in range(len(clients_params[0])):
        assert np.array_equal(own_agg[i_params], aggregated_weights[i_params])
        assert aggregated_weights[i_params].shape == own_agg[i_params].shape

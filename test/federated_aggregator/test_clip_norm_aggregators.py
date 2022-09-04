import numpy as np

from shfl.federated_aggregator import NormClipAggregator
from shfl.federated_aggregator import WeakDPAggregator


def test_aggregated_weights_NormClip():
    num_clients = 10
    num_layers = 5
    tams = [[128, 64], [64, 64], [64, 64], [64, 32], [32, 10]]

    weights = []
    for i in range(num_clients):
        weights.append([np.random.rand(tams[j][0], tams[j][1]) for j in range(num_layers)])

    clients_params = weights

    avgfa = NormClipAggregator(clip=100)
    aggregated_weights = avgfa.aggregate_weights(clients_params)

    clients_params = np.array(weights)
    own_agg = np.array([np.mean(clients_params[:, layer], axis=0) for layer in range(num_layers)])

    for i in range(num_layers):
        assert np.array_equal(own_agg[i], aggregated_weights[i])
    assert len(aggregated_weights) == num_layers


def test_aggregated_weights_WeakDP():
    num_clients = 10
    num_layers = 5
    tams = [[128, 64], [64, 64], [64, 64], [64, 32], [32, 10]]

    np.random.seed(0)
    weights = []
    for i in range(num_clients):
        weights.append([np.random.rand(tams[j][0], tams[j][1]) for j in range(num_layers)])

    clients_params = weights
    np.random.seed(0)
    avgfa = WeakDPAggregator(clip=100)
    aggregated_weights = avgfa.aggregate_weights(clients_params)

    
    np.random.seed(0)
    clients_params = np.array(weights)
    serialized_params = np.array([avgfa._serialize(v) for v in clients_params ])
    for i,v in enumerate(serialized_params):
        serialized_params[i] = v+ np.random.normal(loc=0.0, scale=0.025, size=v.shape)
    clients_params = np.array([avgfa._deserialize(v) for v in serialized_params ])

    own_agg = np.array([np.mean(clients_params[:, layer], axis=0) for layer in range(num_layers)])

    for i in range(num_layers):
        assert np.array_equal(own_agg[i], aggregated_weights[i])
    assert len(aggregated_weights) == num_layers


def test_aggregated_weights_multidimensional_2D_array_NormClip():
    num_clients = 10
    num_rows_params = 3
    num_cols_params = 9

    clients_params = []
    for i in range(num_clients):
        clients_params.append(np.random.rand(num_rows_params, num_cols_params))
    clients_params = np.array(clients_params)

    avgfa = NormClipAggregator(clip=10)
    aggregated_weights = avgfa.aggregate_weights(clients_params)

    own_agg = np.zeros((num_rows_params, num_cols_params))
    for i_client in range(num_clients):
        own_agg += clients_params[i_client]
    own_agg = own_agg / num_clients

    assert np.array_equal(own_agg, aggregated_weights)
    assert aggregated_weights.shape == own_agg.shape


def test_aggregated_weights_multidimensional_2D_array_WeakDP():
    num_clients = 10
    num_rows_params = 3
    num_cols_params = 9

    clients_params = []
    for i in range(num_clients):
        clients_params.append(np.random.rand(num_rows_params, num_cols_params))
    clients_params = np.array(clients_params)

    np.random.seed(0)
    avgfa = WeakDPAggregator(clip=100)
    aggregated_weights = avgfa.aggregate_weights(clients_params)

    np.random.seed(0)
    own_agg = np.zeros((num_rows_params, num_cols_params))
    for v in clients_params:
        own_agg += v + np.random.normal(loc=0.0, scale=0.025, size=v.shape)
    own_agg = own_agg / num_clients

    assert np.array_equal(own_agg, aggregated_weights)
    assert len(aggregated_weights) == own_agg.shape[0]


def test_aggregated_weights_multidimensional_3D_array_NormClip():
    num_clients = 10
    num_rows_params = 3
    num_cols_params = 9
    num_k_params = 5

    clients_params = []
    for i in range(num_clients):
        clients_params.append(np.random.rand(num_rows_params, num_cols_params, num_k_params))
    clients_params = np.array(clients_params)

    avgfa = NormClipAggregator(clip=10)
    aggregated_weights = avgfa.aggregate_weights(clients_params)

    own_agg = np.zeros((num_rows_params, num_cols_params, num_k_params))
    for i_client in range(num_clients):
        own_agg += clients_params[i_client]
    own_agg = own_agg / num_clients

    assert np.array_equal(own_agg, aggregated_weights)
    assert aggregated_weights.shape == own_agg.shape


def test_aggregated_weights_multidimensional_3D_array_WeakDP():
    num_clients = 10
    num_rows_params = 3
    num_cols_params = 9
    num_k_params = 5

    clients_params = []
    for i in range(num_clients):
        clients_params.append(np.random.rand(num_rows_params, num_cols_params, num_k_params))
    clients_params = np.array(clients_params)

    np.random.seed(0)
    avgfa = WeakDPAggregator(clip=10)
    aggregated_weights = avgfa.aggregate_weights(clients_params)

    np.random.seed(0)
    own_agg = np.zeros((num_rows_params, num_cols_params, num_k_params))
    for v in clients_params:
        own_agg += v + np.random.normal(loc=0.0, scale=0.025, size=v.shape)
    own_agg = own_agg / num_clients

    assert np.array_equal(own_agg, aggregated_weights)
    assert len(aggregated_weights) == own_agg.shape[0]


def test_aggregated_weights_list_of_arrays_NormClip():
    num_clients = 10

    clients_params = []
    for i_client in range(num_clients):
        clients_params.append([np.random.rand(30, 20),
                               np.random.rand(20, 30),
                               np.random.rand(50, 40)])

    avgfa = NormClipAggregator(clip=100)
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


def test_aggregated_weights_list_of_arrays_WeakDP():
    num_clients = 10
    seed = 1231231

    clients_params = []
    for i_client in range(num_clients):
        clients_params.append([np.random.rand(30, 20),
                               np.random.rand(20, 30),
                               np.random.rand(50, 40)])
    np.random.seed(seed)
    avgfa = WeakDPAggregator(clip=100)
    aggregated_weights = avgfa.aggregate_weights(clients_params)

    np.random.seed(seed)
    own_agg = [np.zeros((30, 20)),
               np.zeros((20, 30)),
               np.zeros((50, 40))]
    for i_client in range(num_clients):
        for i_params in range(len(clients_params[0])):
            own_agg[i_params] += clients_params[i_client][i_params] 
            own_agg[i_params] += np.random.normal(loc=0.0, scale=0.025, size=clients_params[i_client][i_params].shape)
    for i_params in range(len(clients_params[0])):
        own_agg[i_params] = own_agg[i_params] / num_clients

    for i_params in range(len(clients_params[0])):
        assert np.allclose(own_agg[i_params], aggregated_weights[i_params])
        assert aggregated_weights[i_params].shape == own_agg[i_params].shape


def test_serialization_deserialization_multidimensional_3D_array():
    num_clients = 10
    num_rows_params = 3
    num_cols_params = 9
    num_k_params = 5

    clients_params = []
    for i in range(num_clients):
        clients_params.append(np.random.rand(num_rows_params, num_cols_params, num_k_params))

    avgfa = NormClipAggregator(clip=100)

    serialized_params = np.array([avgfa._serialize(client)  for client in clients_params])
    deserialized_params = np.array([avgfa._deserialize(client) for client in serialized_params])
    
    assert np.array_equal(deserialized_params, clients_params)


def test_serialization_deserialization_multidimensional_2D_array():
    num_clients = 10
    num_rows_params = 3
    num_cols_params = 9

    clients_params = []
    for i in range(num_clients):
        clients_params.append(np.random.rand(num_rows_params, num_cols_params))

    avgfa = NormClipAggregator(clip=100)

    serialized_params = np.array([avgfa._serialize(client)  for client in clients_params])
    deserialized_params = np.array([avgfa._deserialize(client) for client in serialized_params])
    
    assert np.array_equal(deserialized_params, clients_params)


def test_serialization_deserialization():
    num_clients = 10
    num_layers = 5
    tams = [[128, 64], [64, 64], [64, 64], [64, 32], [32, 10]]
    
    weights = []
    for i in range(num_clients):
        weights.append([np.random.rand(tams[j][0], tams[j][1]) for j in range(num_layers)])

    clients_params = weights

    avgfa = NormClipAggregator(clip=100)

    serialized_params = np.array([avgfa._serialize(client)  for client in clients_params])
    for i, client in enumerate(serialized_params):
        deserialized = avgfa._deserialize(client)
        for j, arr in enumerate(deserialized):
            assert np.array_equal(arr,clients_params[i][j])


def test_serialization_deserialization_list_of_arrays():
    num_clients = 10

    clients_params = []
    for i_client in range(num_clients):
        clients_params.append([np.random.rand(30, 20),
                               np.random.rand(20, 30),
                               np.random.rand(50, 40)])

    avgfa = NormClipAggregator(clip=100)

    serialized_params = np.array([avgfa._serialize(client)  for client in clients_params])
    for i, client in enumerate(serialized_params):
        deserialized = avgfa._deserialize(client)
        for j, arr in enumerate(deserialized):
            assert np.array_equal(arr,clients_params[i][j])


def test_serialization_deserialization_mixed_list():
    num_clients = 10

    clients_params = []
    for i_client in range(num_clients):
        clients_params.append([np.random.rand(),
                               np.random.rand(20, 30),
                               np.random.rand(50, 40)])

    avgfa = NormClipAggregator(clip=100)

    serialized_params = np.array([avgfa._serialize(client)  for client in clients_params])
    for i, client in enumerate(serialized_params):
        deserialized = avgfa._deserialize(client)
        for j, arr in enumerate(deserialized):
            assert np.array_equal(arr,clients_params[i][j])

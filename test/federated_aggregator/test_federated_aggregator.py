from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class TestFederatedAggregator(FederatedAggregator):
    def aggregate_weights(self, clients_params):
        pass


def test_federated_aggregator_private_data():
    percentage = 100
    fa = TestFederatedAggregator(percentage)

    assert fa._percentage == percentage


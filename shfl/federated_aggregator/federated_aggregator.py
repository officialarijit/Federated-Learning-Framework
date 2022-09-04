import abc


class FederatedAggregator(abc.ABC):
    """
    Interface for Federated Aggregator.

    # Arguments:
        percentage: Percentage of total data in each client (default None)
    """

    def __init__(self, percentage=None):
        self._percentage = percentage

    @abc.abstractmethod
    def aggregate_weights(self, clients_params):
        """
        Abstract method that aggregates the weights of the client models.

        # Arguments:
            clients_params: Params that represents local clients learning models.
        # Returns:
            aggregated_weights: Aggregated weights
        """

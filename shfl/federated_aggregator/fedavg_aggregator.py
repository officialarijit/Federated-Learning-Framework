import numpy as np

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic


class FedAvgAggregator(FederatedAggregator):
    """
    Implementation of Average Federated Aggregator.
    It only uses a simple average of the parameters of all the models.

    It implements [Federated Aggregator](../federated_aggregator/#federatedaggregator-class)
    """

    def aggregate_weights(self, clients_params):
        """
        Implementation of abstract method of class
        [AggregateWeightsFunction](../federated_aggregator/#federatedaggregator-class)
        # Arguments:
            clients_params: list of multi-dimensional (numeric) arrays.
            Each entry in the list contains the model's parameters of one client.

        # Returns
            aggregated_weights: aggregated weights representing the global learning model

        # References
            [Communication-Efficient Learning of Deep Networks
            from Decentralized Data](https://arxiv.org/abs/1602.05629)
        """

        return self._aggregate(*clients_params)

    @dispatch(Variadic[np.ndarray, np.ScalarType])
    def _aggregate(self, *params):
        """Aggregation of arrays"""
        return np.mean(np.array(params), axis=0)

    @dispatch(Variadic[list])
    def _aggregate(self, *params):
        """Aggregation of (nested) lists of arrays"""
        aggregated_weights = [self._aggregate(*params)
                              for params in zip(*params)]
        return aggregated_weights

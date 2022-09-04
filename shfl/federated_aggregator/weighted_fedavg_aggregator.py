import numpy as np
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class WeightedFedAvgAggregator(FederatedAggregator):
    """
    Implementation of Weighted Federated Averaging Aggregator.
    The aggregation of the parameters is weighted by the number of data \
    in every node.

    It implements [Federated Aggregator](../federated_aggregator/#federatedaggregator-class)
    """

    def aggregate_weights(self, clients_params):
        """
        Implementation of abstract method of class
        [AggregateWeightsFunction](../federated_aggregator/#federatedaggregator-class)

        # Arguments:
            clients_params: list of multi-dimensional (numeric) arrays.
            Each entry in the list contains the model's parameters of one client.

        # Returns:
            aggregated_weights: aggregator weights representing the global learning model
        """
        ponderated_weights = [self._ponderate_weights(i_client, i_weight)
                              for i_client, i_weight
                              in zip(clients_params, self._percentage)]

        return self._aggregate(*ponderated_weights)

    @dispatch((np.ndarray, np.ScalarType), np.ScalarType)
    def _ponderate_weights(self, params, weight):
        """Weighting of arrays"""
        return params * weight

    @dispatch(list, np.ScalarType)
    def _ponderate_weights(self, params, weight):
        """Weighting of (nested) lists of arrays"""
        ponderated_weights = [self._ponderate_weights(i_params, weight)
                              for i_params in params]
        return ponderated_weights

    @dispatch(Variadic[np.ndarray, np.ScalarType])
    def _aggregate(self, *ponderated_weights):
        """Aggregation of ponderated arrays"""
        return np.sum(np.array(ponderated_weights), axis=0)

    @dispatch(Variadic[list])
    def _aggregate(self, *ponderated_weights):
        """Aggregation of ponderated (nested) lists of arrays"""
        aggregated_weights = [self._aggregate(*params)
                              for params in zip(*ponderated_weights)]
        return aggregated_weights

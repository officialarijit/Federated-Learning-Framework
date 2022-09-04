import numpy as np
from numpy import linalg as LA

from shfl.federated_aggregator import FederatedAggregator
from multipledispatch import dispatch
from multipledispatch.variadic import Variadic


class NormClipAggregator(FederatedAggregator):
    
    def __init__(self, clip):
        self._clip = clip
    
    def aggregate_weights(self, clients_params):
        return self._aggregate(*clients_params)

    def _serialize(self, data):
        data = [np.array(j) for j in data]
        self._data_shape_list = [j.shape for j in data]
        serialized_data = [j.ravel() for j in data]
        serialized_data = np.hstack(serialized_data)
        return serialized_data
        
    def _deserialize(self, data):
        firstInd = 0
        deserialized_data = []
        for shp in self._data_shape_list:
            if len(shp) > 1:
                shift = np.prod(shp)
            elif len(shp) == 0:
                shift = 1
            else:
                shift = shp[0]
            tmp_array = data[firstInd:firstInd+shift]
            tmp_array = tmp_array.reshape(shp)
            deserialized_data.append(tmp_array)
            firstInd += shift
        return deserialized_data

    @dispatch(Variadic[np.ndarray, np.ScalarType])
    def _aggregate(self, *params):
        """Aggregation of arrays"""
        clients_params = np.array(params)
        for i, v in enumerate(clients_params):
            norm = LA.norm(v)
            clients_params[i] = np.multiply(v, min(1, self._clip/norm))
        
        return np.mean(clients_params, axis=0)

    @dispatch(Variadic[list])
    def _aggregate(self, *params):
        """Aggregation of (nested) lists of arrays"""        
        serialized_params = np.array([self._serialize(client) for client in params])
        serialized_aggregation = self._aggregate(*serialized_params)
        aggregated_weights = self._deserialize(serialized_aggregation)
        
        return aggregated_weights

class WeakDPAggregator(NormClipAggregator):

    @dispatch(Variadic[np.ndarray, np.ScalarType])
    def _aggregate(self, *params):
        """Aggregation of arrays"""
        clients_params = np.array(params)
        for i, v in enumerate(clients_params):
            norm = LA.norm(v)
            clients_params[i] = np.multiply(v, min(1, self._clip/norm))
            clients_params[i] += np.random.normal(loc=0.0, scale=0.025, size=v.shape) 
        
        return np.mean(clients_params, axis=0)

    @dispatch(Variadic[list])
    def _aggregate(self, *params):
        """Aggregation of (nested) lists of arrays"""        
        serialized_params = np.array([self._serialize(client) for client in params])
        serialized_aggregation = self._aggregate(*serialized_params)
        aggregated_weights = self._deserialize(serialized_aggregation)
        
        return aggregated_weights
    

import numpy as np

from shfl.data_base.data_base import shuffle_rows
from shfl.data_distribution.data_distribution_sampling import SamplingDataDistribution


class IidDataDistribution(SamplingDataDistribution):
    """
    Implementation of an independent and identically distributed data distribution using \
        [Data Distribution](../data_distribution/#datadistribution-class)
    """

    def make_data_federated(self, data, labels, percent, num_nodes=1, weights=None, sampling="without_replacement"):
        """
        Method that makes data and labels argument federated in an iid scenario.
        The data and labels may be numpy arrays or pandas dataframe/series.

        # Arguments:
            data: Data to federate
            labels: Labels to federate
            num_nodes: Number of nodes to create
            percent: Percent of the data (between 0 and 100) to be distributed
            weights: Array of weights for weighted distribution (default is None)
            sampling: methodology between with or without sampling (default "without_sampling")

        # Returns:
            federated_data: A list containing the data for each client
            federated_label: A list containing the labels for each client
        """
        if weights is None:
            weights = np.full(num_nodes, 1/num_nodes)

        # Shuffle data
        data, labels = shuffle_rows(data, labels)

        # Select percent
        data = data[0:int(percent * len(data) / 100)]
        labels = labels[0:int(percent * len(labels) / 100)]

        federated_data = []
        federated_label = []

        if sampling == "without_replacement":
            if sum(weights) > 1:
                weights = np.array([float(i)/sum(weights) for i in weights])

            sum_used = 0
            percentage_used = 0

            for client in range(0, num_nodes):
                federated_data.append(data[sum_used:int((percentage_used + weights[client]) * len(data))])
                federated_label.append(labels[sum_used:int((percentage_used + weights[client]) * len(labels))])

                sum_used = int((percentage_used + weights[client]) * len(data))
                percentage_used += weights[client]
        else:
            for client in range(0, num_nodes):
                federated_data.append(data[:int((weights[client]) * len(data))])
                federated_label.append(labels[:int((weights[client]) * len(labels))])

                data, labels = shuffle_rows(data, labels)

        if isinstance(data, np.ndarray) and isinstance(labels, np.ndarray):
            federated_data = np.array(federated_data)
            federated_label = np.array(federated_label)

        return federated_data, federated_label

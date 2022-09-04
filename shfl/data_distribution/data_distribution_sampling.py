import abc

from shfl.data_distribution.data_distribution import DataDistribution


class SamplingDataDistribution(DataDistribution):
    """
    Abstract class for a sampling data distribution using \
        [Data Distribution](../data_distribution/#datadistribution-class)
    """

    @abc.abstractmethod
    def make_data_federated(self, data, labels, percent, num_nodes=1, weights=None, sampling="without_sampling"):
        """
        Method that must implement every data distribution extending this class
        # Arguments:
            data: Array of data
            labels: Labels
            percent: Percent of the data (between 0 and 100) to be distributed (default is 100)
            num_nodes : Number of nodes (default is 1)
            weights: Array of weights for weighted distribution (default is None)
            sampling: methodology between with or without sampling (default "without_sampling")
        # Returns:
            federated_data: Data for each client
            federated_label: Labels for each client
        """

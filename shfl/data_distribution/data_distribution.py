import abc

from shfl.private.data import LabeledData
from shfl.private.federated_operation import FederatedData


class DataDistribution(abc.ABC):
    """
    Abstract class for data distribution

    # Arguments:
        database: Database to distribute. (see: [Databases](../databases))
    """

    def __init__(self, database):
        self._database = database

    def get_federated_data(self, percent=100, *args, **kwargs):
        """
        Method that splits the whole data between the established number of nodes.

        # Arguments:
            num_nodes: Number of nodes to create
            percent: Percent of the data (between 0 and 100) to be distributed (default is 100)

        # Returns:
              * **federated_data, test_data, test_label**
        """

        train_data, train_label = self._database.train
        test_data, test_label = self._database.test

        federated_train_data, federated_train_label = self.make_data_federated(train_data,
                                                                               train_label,
                                                                               percent,
                                                                               *args, **kwargs)

        federated_data = FederatedData()
        num_nodes = len(federated_train_label)
        for node in range(num_nodes):
            node_data = LabeledData(federated_train_data[node], federated_train_label[node])
            federated_data.add_data_node(node_data)

        return federated_data, test_data, test_label

    @abc.abstractmethod
    def make_data_federated(self, data, labels, percent, *args, **kwargs):
        """
        Method that must implement every data distribution extending this class

        # Arguments:
            data: Array of data
            labels: Labels
            percent: Percent of the data (between 0 and 100) to be distributed (default is 100)

        # Returns:
            federated_data: A list containing the data for each client
            federated_label: A list containing the labels for each client
        """
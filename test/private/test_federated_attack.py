import numpy as np

from shfl.private.federated_operation import FederatedData
from shfl.private.data import LabeledData
from shfl.private.federated_attack import ShuffleNode
from shfl.private.federated_attack import FederatedPoisoningDataAttack
from shfl.private.data import UnprotectedAccess


def test_shuffle_node():
    data = np.random.rand(50).reshape([10, 5])
    label = np.random.randint(0, 10, 10)
    labeled_data = LabeledData(data, label)

    federated_data = FederatedData()
    federated_data.add_data_node(labeled_data)
    for node in federated_data:
        node.apply_data_transformation(ShuffleNode())

    federated_data.configure_data_access(UnprotectedAccess())
    assert (not np.array_equal(federated_data[0].query().label, label))


def test_federated_poisoning_attack():
    num_nodes = 10
    federated_data = FederatedData()

    list_labels = []
    for i in range(num_nodes):
        data = np.random.rand(50).reshape([10, 5])
        label = np.random.randint(0, 10, 10)
        list_labels.append(label)
        labeled_data = LabeledData(data, label)
        federated_data.add_data_node(labeled_data)

    percentage = 10
    simple_attack = FederatedPoisoningDataAttack(percentage=percentage)
    simple_attack.apply_attack(federated_data=federated_data)

    adversaries_idx = simple_attack.adversaries

    federated_data.configure_data_access(UnprotectedAccess())
    for node, idx in zip(federated_data, range(num_nodes)):
        if idx in adversaries_idx:
            assert not np.array_equal(node.query().label, list_labels[idx])
        else:
            assert np.array_equal(node.query().label, list_labels[idx])


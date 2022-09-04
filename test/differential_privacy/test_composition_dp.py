import pytest
import numpy as np

from shfl.private import DataNode
from shfl.private.data import UnprotectedAccess
from shfl.differential_privacy.composition_dp import ExceededPrivacyBudgetError
from shfl.differential_privacy.composition_dp import AdaptiveDifferentialPrivacy
from shfl.differential_privacy.dp_mechanism import GaussianMechanism


def test_exception__budget():
    exception = ExceededPrivacyBudgetError(epsilon_delta=1)
    assert str(exception) is not None


def test_exception_exceeded_privacy_budget_error():
    scalar = 175

    dp_mechanism = GaussianMechanism(1, epsilon_delta=(0.1, 1))
    data_access_definition = AdaptiveDifferentialPrivacy(epsilon_delta=(1, 0),
                                                         differentially_private_mechanism=dp_mechanism)
    node = DataNode()
    node.set_private_data("scalar", scalar)
    node.configure_data_access("scalar", data_access_definition)

    with pytest.raises(ExceededPrivacyBudgetError):
        node.query("scalar")


def test_constructor_bad_params():
    with pytest.raises(ValueError):
        AdaptiveDifferentialPrivacy(epsilon_delta=(1, 2, 3))

    with pytest.raises(ValueError):
        AdaptiveDifferentialPrivacy(epsilon_delta=(-1, 2))

    with pytest.raises(ValueError):
        AdaptiveDifferentialPrivacy(epsilon_delta=(1, -2))

    with pytest.raises(ValueError):
        AdaptiveDifferentialPrivacy(epsilon_delta=(1, 1), differentially_private_mechanism=UnprotectedAccess())


def test_configure_data_access():
    data_access_definition = AdaptiveDifferentialPrivacy(epsilon_delta=(1, 1))
    data_node = DataNode()
    data_node.set_private_data("test", np.array(range(10)))
    with pytest.raises(ValueError):
        data_node.configure_data_access("test", data_access_definition)
        data_node.query("test")


def test_data_access():
    data_access_definition = AdaptiveDifferentialPrivacy(epsilon_delta=(1, 1))
    data_node = DataNode()
    array = np.array(range(10))
    data_node.set_private_data("test", array)

    data_node.configure_data_access("test", data_access_definition)
    query_result = data_node.query("test", differentially_private_mechanism=GaussianMechanism(1,
                                                                                              epsilon_delta=(0.1, 1)))

    assert query_result is not None


def test_exception_no_access_definition():
    data_access_definition = AdaptiveDifferentialPrivacy(epsilon_delta=(1, 1))
    data_node = DataNode()
    array = np.array(range(10))
    data_node.set_private_data("test", array)

    data_node.configure_data_access("test", data_access_definition)
    with pytest.raises(ValueError):
        data_node.query("test")


def test_exception_budget():
    dp_mechanism = GaussianMechanism(1, epsilon_delta=(0.1, 1))
    data_access_definition = AdaptiveDifferentialPrivacy(epsilon_delta=(1, 1),
                                                         differentially_private_mechanism=dp_mechanism)
    data_node = DataNode()
    array = np.array(range(10))
    data_node.set_private_data("test", array)

    data_node.configure_data_access("test", data_access_definition)
    with pytest.raises(ExceededPrivacyBudgetError):
        for i in range(1, 1000):
            data_node.query("test")


def test_exception_budget_2():
    data_access_definition = AdaptiveDifferentialPrivacy(epsilon_delta=(1, 0.001))
    data_node = DataNode()
    array = np.array(range(10))
    data_node.set_private_data("test", array)

    data_node.configure_data_access("test", data_access_definition)
    with pytest.raises(ExceededPrivacyBudgetError):
        for i in range(1, 1000):
            data_node.query("test", differentially_private_mechanism=GaussianMechanism(1, epsilon_delta=(0.1, 1)))

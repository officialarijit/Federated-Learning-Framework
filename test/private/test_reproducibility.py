from shfl.private import Reproducibility
import numpy as np
import pytest


def test_reproducibility():
    Reproducibility.get_instance().delete_instance()

    seed = 1234
    Reproducibility(seed)

    assert Reproducibility.get_instance().seed == seed
    assert Reproducibility.get_instance().seeds['server'] == seed
    assert np.random.get_state()[1][0] == seed


def test_reproducibiliry_singleton():
    Reproducibility.get_instance().delete_instance()

    seed = 1234
    Reproducibility(seed)

    with pytest.raises(Exception):
        Reproducibility()


def test_set_seed():
    Reproducibility.get_instance().delete_instance()

    seed = 1234
    Reproducibility(seed)

    id = 'ID0'
    Reproducibility.get_instance().set_seed(id)

    assert Reproducibility.get_instance().seeds[id]

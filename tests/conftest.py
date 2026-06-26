import pytest

from complex_tokenization.graphs.settings import GraphSettings
from complex_tokenization.graphs.units import _cluster_handlers


@pytest.fixture(autouse=True)
def reset_graph_settings():
    original = {
        "MAX_MERGE_SIZE": GraphSettings.MAX_MERGE_SIZE,
        "ONLY_MINIMAL_MERGES": GraphSettings.ONLY_MINIMAL_MERGES,
        "TRADE_MEMORY_FOR_SPEED": GraphSettings.TRADE_MEMORY_FOR_SPEED,
    }
    yield
    GraphSettings.MAX_MERGE_SIZE = original["MAX_MERGE_SIZE"]
    GraphSettings.ONLY_MINIMAL_MERGES = original["ONLY_MINIMAL_MERGES"]
    GraphSettings.TRADE_MEMORY_FOR_SPEED = original["TRADE_MEMORY_FOR_SPEED"]


@pytest.fixture(autouse=True)
def clear_script_registry():
    _cluster_handlers.clear()
    yield
    _cluster_handlers.clear()

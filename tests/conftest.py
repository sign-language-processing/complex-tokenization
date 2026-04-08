import pytest

from complex_tokenization.graphs.settings import GraphSettings


@pytest.fixture(autouse=True)
def reset_graph_settings():
    original = {
        "USE_SINGLETONS": GraphSettings.USE_SINGLETONS,
        "MAX_MERGE_SIZE": GraphSettings.MAX_MERGE_SIZE,
        "ONLY_MINIMAL_MERGES": GraphSettings.ONLY_MINIMAL_MERGES,
    }
    yield
    GraphSettings.USE_SINGLETONS = original["USE_SINGLETONS"]
    GraphSettings.MAX_MERGE_SIZE = original["MAX_MERGE_SIZE"]
    GraphSettings.ONLY_MINIMAL_MERGES = original["ONLY_MINIMAL_MERGES"]


@pytest.fixture(autouse=True)
def clear_singleton_cache():
    from complex_tokenization.graph import GraphVertex
    GraphVertex._instances.clear()
    yield
    GraphVertex._instances.clear()

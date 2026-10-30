import pytest

from rag import config


@pytest.fixture(autouse=True)
def default_to_dense_search(monkeypatch):
    """Tests must not depend on which SEARCH_MODE you chose in rag/config.py."""
    monkeypatch.setattr(config, "SEARCH_MODE", "dense")

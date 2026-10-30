from rag.store import VectorStore

from .helpers import CHUNKS, HashingEmbedder


def test_search_finds_the_relevant_chunk():
    store = VectorStore.build(HashingEmbedder(), CHUNKS)
    top = store.search("Which model had the highest accuracy for migraine classification?", k=2, mode="dense")
    assert top[0]["id"] == 0
    assert top[0]["score"] >= top[1]["score"]


def test_save_and_load_round_trip(tmp_path):
    embedder = HashingEmbedder()
    VectorStore.build(embedder, CHUNKS).save(tmp_path)
    loaded = VectorStore.load(tmp_path, embedder)
    assert len(loaded.chunks) == 3
    assert loaded.search("class imbalance SMOTE", k=1, mode="dense")[0]["id"] == 1


def test_loading_missing_index_gives_helpful_error(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="ingest.py"):
        VectorStore.load(tmp_path, HashingEmbedder())


def test_k_larger_than_corpus_is_safe():
    store = VectorStore.build(HashingEmbedder(), CHUNKS)
    assert len(store.search("anything", k=50, mode="dense")) == 3


def test_keyword_mode_finds_exact_terms():
    store = VectorStore.build(HashingEmbedder(), CHUNKS)
    top = store.search("heart rate variability wearable", k=1, mode="keyword")
    assert top[0]["id"] == 2


def test_hybrid_mode_returns_k_distinct_results_including_best_match():
    store = VectorStore.build(HashingEmbedder(), CHUNKS)
    top = store.search("SMOTE class imbalance", k=2, mode="hybrid")
    assert len(top) == 2 and len({r["id"] for r in top}) == 2
    assert top[0]["id"] == 1


def test_unknown_search_mode_is_rejected():
    import pytest

    store = VectorStore.build(HashingEmbedder(), CHUNKS)
    with pytest.raises(ValueError, match="Unknown search mode"):
        store.search("anything", mode="magic")


def test_keyword_mode_never_returns_fewer_than_k_passages():
    store = VectorStore.build(HashingEmbedder(), CHUNKS)
    results = store.search("zzzz qqqq unrelated words", k=2, mode="keyword")
    assert len(results) == 2 and len({r["id"] for r in results}) == 2

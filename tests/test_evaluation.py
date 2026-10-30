from rag.evaluation import hit_at_k
from rag.store import VectorStore

from .helpers import CHUNKS, HashingEmbedder


def test_hit_rate_counts_hits_and_misses():
    store = VectorStore.build(HashingEmbedder(), CHUNKS)
    questions = [
        {"question": "Which model had the highest accuracy?", "expected_phrase": "XGBoost"},
        {"question": "How was class imbalance handled?", "expected_phrase": "SMOTE"},
        {"question": "What is the capital of France?", "expected_phrase": "Paris"},
    ]
    result = hit_at_k(store, questions, k=1, mode="dense")
    assert result["total"] == 3 and result["hits"] == 2
    assert abs(result["hit_rate"] - 2 / 3) < 1e-9


def test_expected_phrases_list_counts_any_matching_phrase():
    store = VectorStore.build(HashingEmbedder(), CHUNKS)
    questions = [
        {"question": "Which model had the highest accuracy?",
         "expected_phrases": ["nothing like this", "XGBoost"]},
        {"question": "What is the capital of France?", "expected_phrases": ["Paris", "Lyon"]},
    ]
    result = hit_at_k(store, questions, k=1, mode="dense")
    assert result["hits"] == 1
    assert result["details"][1]["top_passages"][0]["source"]

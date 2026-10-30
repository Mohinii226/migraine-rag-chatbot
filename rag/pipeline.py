"""Glue: question -> retrieve passages -> generate answer."""
from . import config
from .embeddings import SentenceTransformerEmbedder
from .llm import generate_answer
from .store import VectorStore


def load_store():
    embedder = SentenceTransformerEmbedder(config.EMBED_MODEL)
    return VectorStore.load(config.INDEX_DIR, embedder)


def answer_question(store, question, k=config.TOP_K):
    passages = store.search(question, k=k)
    result = generate_answer(question, passages)
    result["passages"] = passages
    return result

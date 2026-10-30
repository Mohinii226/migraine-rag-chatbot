import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "data" / "papers"   # put your PDFs here
INDEX_DIR = ROOT / "index"              # the searchable index is saved here

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, free, runs on your laptop
CHUNK_SIZE = 900      # characters per chunk (roughly 150 words)
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks
TOP_K = 4             # how many passages to retrieve for each question
SEARCH_MODE = "hybrid"  # "dense" (embeddings), "keyword" (TF-IDF) or "hybrid" (both, fused)

# Claude model used to write answers. Haiku is the cheapest. You can override it
# by setting ANTHROPIC_MODEL in your .env file (see Anthropic's docs for model names).
DEFAULT_LLM_MODEL = "claude-haiku-4-5-20251001"


def llm_model() -> str:
    return os.getenv("ANTHROPIC_MODEL", DEFAULT_LLM_MODEL)

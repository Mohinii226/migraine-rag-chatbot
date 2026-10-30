"""Run this FIRST (and again whenever you add or remove papers).

    python ingest.py

It reads every PDF/TXT in data/papers/, cuts the text into chunks, turns the chunks
into embeddings, and saves a searchable index into the index/ folder.
"""
import sys

from rag import config
from rag.chunking import make_chunks
from rag.loader import load_documents
from rag.store import VectorStore


def ingest(papers_dir, index_dir, embedder, chunk_size=config.CHUNK_SIZE,
           overlap=config.CHUNK_OVERLAP, show_progress=False):
    pages = load_documents(papers_dir)
    if not pages:
        raise SystemExit(
            f"No documents found in {papers_dir}.\n"
            "Copy some PDF files into that folder and run this command again.")
    chunks = make_chunks(pages, chunk_size, overlap)
    sources = sorted({p["source"] for p in pages})
    print(f"Read {len(pages)} pages from {len(sources)} file(s): {', '.join(sources)}")
    print(f"Created {len(chunks)} chunks. Embedding them (the first run downloads a small model)...")
    store = VectorStore.build(embedder, chunks, show_progress=show_progress)
    store.save(index_dir)
    print(f"Done. Saved the index to {index_dir} using the '{store.backend}' backend.")
    if store.backend != "faiss":
        print("Note: FAISS is not installed, so NumPy search was used. "
              "Run `pip install faiss-cpu` to use a real FAISS index.")
    return store


def main():
    from rag.embeddings import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(config.EMBED_MODEL)
    ingest(config.PAPERS_DIR, config.INDEX_DIR, embedder, show_progress=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

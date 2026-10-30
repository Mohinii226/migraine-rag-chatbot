"""Step 2 of RAG: cut long pages into small overlapping pieces ("chunks").

Why? An LLM prompt has limited room, and a search works best on focused passages.
Overlap keeps a sentence that sits on a boundary from losing its context.
"""
import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def clean_text(text):
    text = text.replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join words hyphenated across lines
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text(text, chunk_size=900, overlap=150):
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")
    text = clean_text(text)
    if not text:
        return []

    sentences = []
    for sentence in _SENTENCE_SPLIT.split(text):
        while len(sentence) > chunk_size:  # very long "sentence": cut it
            sentences.append(sentence[:chunk_size])
            sentence = sentence[chunk_size:]
        if sentence:
            sentences.append(sentence)

    chunks, current = [], []
    length = 0
    for sentence in sentences:
        added = len(sentence) + (1 if current else 0)
        if current and length + added > chunk_size:
            chunks.append(" ".join(current))
            keep, kept_len = [], 0  # carry the last few sentences into the next chunk
            for previous in reversed(current):
                if kept_len + len(previous) + 1 > overlap:
                    break
                keep.insert(0, previous)
                kept_len += len(previous) + 1
            current = keep
            length = len(" ".join(current))
            added = len(sentence) + (1 if current else 0)
        current.append(sentence)
        length += added
    if current:
        chunks.append(" ".join(current))
    return chunks


def make_chunks(pages, chunk_size=900, overlap=150):
    """Turn pages into chunk dicts: {"id", "text", "source", "page"}."""
    chunks = []
    for page in pages:
        for piece in split_text(page["text"], chunk_size, overlap):
            chunks.append({"id": len(chunks), "text": piece,
                           "source": page["source"], "page": page["page"]})
    return chunks

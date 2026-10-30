import pytest

from rag.chunking import clean_text, make_chunks, split_text


def test_empty_text_gives_no_chunks():
    assert split_text("   \n  ") == []


def test_chunks_respect_size_and_cover_all_sentences():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = split_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 3
    assert all(len(c) <= 200 + 50 for c in chunks)
    joined = " ".join(chunks)
    for i in range(60):
        assert f"Sentence number {i} is here." in joined


def test_neighbouring_chunks_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = split_text(text, chunk_size=200, overlap=80)
    last_sentence_of_first = chunks[0].split(". ")[-1].strip(".")
    assert last_sentence_of_first[:20] in chunks[1]


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        split_text("Hello world.", chunk_size=100, overlap=100)


def test_very_long_sentence_is_cut():
    chunks = split_text("a" * 2500, chunk_size=900, overlap=100)
    assert len(chunks) >= 3


def test_hyphenated_line_breaks_are_joined():
    assert clean_text("classi-\nfication works") == "classification works"


def test_make_chunks_keeps_source_and_page():
    pages = [{"text": "One. Two. Three.", "source": "x.pdf", "page": 7}]
    chunks = make_chunks(pages)
    assert chunks[0]["source"] == "x.pdf" and chunks[0]["page"] == 7 and chunks[0]["id"] == 0

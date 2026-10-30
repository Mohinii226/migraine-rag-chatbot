from rag.prompt import SYSTEM_PROMPT, build_prompt

from .helpers import CHUNKS


def test_prompt_numbers_passages_and_includes_sources():
    prompt = build_prompt("What did XGBoost achieve?", CHUNKS[:2])
    assert "[1] (source: paper_a.pdf, page 1)" in prompt
    assert "[2] (source: paper_a.pdf, page 2)" in prompt
    assert "Question: What did XGBoost achieve?" in prompt


def test_system_prompt_restricts_to_context():
    assert "ONLY" in SYSTEM_PROMPT and "medical advice" in SYSTEM_PROMPT

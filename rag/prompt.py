"""Step 5 of RAG: build the prompt that gives the LLM the retrieved passages."""

SYSTEM_PROMPT = (
    "You are a research assistant that answers questions about a set of migraine research "
    "papers. Use ONLY the numbered context passages provided. After each claim, cite the "
    "passage number in square brackets, like [1] or [2][3]. If the passages do not contain "
    "the answer, say 'I could not find this in the provided papers.' Do not use outside "
    "knowledge, and do not give medical advice or diagnoses."
)


def build_prompt(question, passages):
    blocks = []
    for number, passage in enumerate(passages, start=1):
        blocks.append(
            f"[{number}] (source: {passage['source']}, page {passage['page']})\n{passage['text']}")
    context = "\n\n".join(blocks)
    return (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely using only the context passages above, with citations."
    )

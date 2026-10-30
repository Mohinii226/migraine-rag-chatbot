"""Measure retrieval quality with hit@k.

Each test question lists one or more phrases that prove a retrieved passage contains the
answer. A question is a "hit" if ANY of the top-k retrieved chunks contains ANY of its
phrases. Use "expected_phrases" (a list) when more than one paper, or more than one wording,
can answer the question; the older single "expected_phrase" key still works.
"""


def _phrases(item):
    if "expected_phrases" in item:
        return [p.lower() for p in item["expected_phrases"]]
    return [item["expected_phrase"].lower()]


def hit_at_k(store, questions, k=4, mode=None):
    details = []
    for item in questions:
        results = store.search(item["question"], k=k, mode=mode)
        phrases = _phrases(item)
        hit = any(p in r["text"].lower() for r in results for p in phrases)
        details.append({
            "question": item["question"],
            "hit": hit,
            "expected": phrases,
            "top_passages": [{"source": r["source"], "page": r["page"],
                              "snippet": r["text"][:110].replace("\n", " ")} for r in results],
        })
    hits = sum(d["hit"] for d in details)
    return {"k": k, "mode": mode, "total": len(details), "hits": hits,
            "hit_rate": hits / len(details) if details else 0.0, "details": details}

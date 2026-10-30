"""Check your evaluation questions BEFORE measuring.

    python check_eval_questions.py

For each question in eval_questions.json it checks that at least one expected phrase really
appears in some chunk of YOUR index. A question whose phrases are all missing would count as a
"miss" in evaluate.py even though the search did nothing wrong.
"""
import json
import sys
from pathlib import Path

from rag import config


def main():
    chunks_file = config.INDEX_DIR / "chunks.json"
    if not chunks_file.exists():
        print("No index found. Run `python ingest.py` first.")
        return 1
    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    questions = json.loads(Path("eval_questions.json").read_text(encoding="utf-8"))

    problems = 0
    for item in questions:
        phrases = item.get("expected_phrases") or [item["expected_phrase"]]
        found = {p: sum(p.lower() in c["text"].lower() for c in chunks) for p in phrases}
        usable = any(found.values())
        problems += 0 if usable else 1
        flag = "ok       " if usable else "NOT FOUND"
        detail = ", ".join(f"{p!r}x{n}" for p, n in found.items())
        print(f"{flag} {item['question'][:62]:62} {detail}")
    print(f"\n{len(questions) - problems} of {len(questions)} questions are usable on your index.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

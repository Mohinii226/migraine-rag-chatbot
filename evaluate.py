"""Measure how well the search step finds the passage that contains the answer (hit@k).

    python evaluate.py                    # uses the search mode set in rag/config.py
    python evaluate.py --verbose          # also shows what was retrieved for every MISS
    python evaluate.py --compare          # compares dense, keyword and hybrid search

eval_questions.json holds the test questions. Each has "expected_phrases": phrases that
prove a retrieved passage contains the answer (from any paper that answers it).
"""
import argparse
import json
import sys
from pathlib import Path

from rag import config
from rag.evaluation import hit_at_k
from rag.pipeline import load_store


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["dense", "keyword", "hybrid"], default=config.SEARCH_MODE)
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    questions = json.loads(Path("eval_questions.json").read_text(encoding="utf-8"))
    store = load_store()

    if args.compare:
        print(f"Comparing search modes (k={args.k}, {len(questions)} questions)\n")
        for mode in ("dense", "keyword", "hybrid"):
            r = hit_at_k(store, questions, k=args.k, mode=mode)
            print(f"{mode:8s} hit@{args.k}: {r['hits']:2d}/{r['total']} = {r['hit_rate']:.0%}")
        return 0

    result = hit_at_k(store, questions, k=args.k, mode=args.mode)
    for d in result["details"]:
        print(("HIT  " if d["hit"] else "MISS ") + d["question"])
        if args.verbose and not d["hit"]:
            for n, p in enumerate(d["top_passages"], start=1):
                print(f"       [{n}] {p['source'][:26]} p{p['page']}: {p['snippet']}")
            print(f"       looking for any of: {d['expected']}")
    print(f"\nmode={args.mode}  hit@{args.k}: {result['hits']}/{result['total']} = {result['hit_rate']:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

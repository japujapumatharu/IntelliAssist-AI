"""Retrieval evaluation -> feeds the Evaluation Report deliverable.

Usage:  python eval/evaluate.py sample_docs/ai_overview.txt eval/eval_set.json
Metric: Hit@k = fraction of questions where at least one expected keyword
appears in the top-k retrieved chunks; plus MRR over the same criterion.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loader import load_file  # noqa: E402
from src.rag import RAGEngine  # noqa: E402


def main(doc_path: str, eval_path: str, ks=(1, 3, 5)):
    p = Path(doc_path)
    engine = RAGEngine()
    n = engine.index(load_file(p.name, p.read_bytes()))
    items = json.loads(Path(eval_path).read_text())
    print(f"Indexed {n} chunks | {len(items)} questions\n")

    ranks = []
    for it in items:
        res = engine.retrieve(it["question"], k=max(ks))
        rank = next(
            (i for i, (d, _) in enumerate(res, 1)
             if any(kw.lower() in d.page_content.lower() for kw in it["keywords"])),
            None,
        )
        ranks.append(rank)
        print(f"  rank={rank}  {it['question']}")

    print()
    for k in ks:
        hit = sum(1 for r in ranks if r and r <= k) / len(ranks)
        print(f"Hit@{k}: {hit:.2f}")
    print(f"MRR:   {sum(1 / r for r in ranks if r) / len(ranks):.2f}")


if __name__ == "__main__":
    main(*(sys.argv[1:3] or ["sample_docs/ai_overview.txt", "eval/eval_set.json"]))

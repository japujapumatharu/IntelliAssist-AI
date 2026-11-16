import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loader import load_file
from src.rag import RAGEngine


def run_eval(doc_path, eval_path, k_values=(1, 3, 5)):
    doc_file = Path(doc_path)
    engine = RAGEngine()
    chunks_count = engine.index(load_file(doc_file.name, doc_file.read_bytes()))
    questions = json.loads(Path(eval_path).read_text(encoding="utf-8"))

    print(f"Indexed {chunks_count} chunks. Running {len(questions)} test queries...\n")

    ranks = []
    for item in questions:
        results = engine.retrieve(item["question"], k=max(k_values))
        matched_rank = None
        for rank_idx, (doc, _) in enumerate(results, start=1):
            text_lower = doc.page_content.lower()
            if any(kw.lower() in text_lower for kw in item["keywords"]):
                matched_rank = rank_idx
                break
        ranks.append(matched_rank)
        print(f"Rank: {matched_rank} | Question: {item['question']}")

    print()
    for k in k_values:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        print(f"Hit@{k}: {hits / len(ranks):.2f}")

    mrr = sum(1.0 / r for r in ranks if r is not None) / len(ranks)
    print(f"MRR:   {mrr:.2f}")


if __name__ == "__main__":
    doc_arg = sys.argv[1] if len(sys.argv) > 1 else "sample_docs/ai_overview.txt"
    eval_arg = sys.argv[2] if len(sys.argv) > 2 else "eval/eval_set.json"
    run_eval(doc_arg, eval_arg)

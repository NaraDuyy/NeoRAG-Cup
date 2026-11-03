"""Chấm `retrieved_ids` cuối cùng của HybridPipeline (sau khi LLM chọn đoạn) trên tập train.

Khác `eval_retrieval.py` (chỉ chấm thứ hạng RRF), script này chạy cả `HybridPipeline.run()`.
Khi prompt không đổi, mọi lần gọi LLM đều trúng cache nên không tốn API:

    python scripts/eval_final_retrieval.py
    python scripts/eval_final_retrieval.py --show   # in các câu còn thiếu đoạn đúng trong top 5

Chia theo nửa `tune` / `hold` của `eval_split.split_train()`.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from neorag.data import DATA_DIR, load_corpus, read_jsonl
from neorag.metrics.retrieval import ndcg_at_k, recall_at_k, reciprocal_rank_at_k
from neorag.pipelines.hybrid import HybridPipeline

sys.path.insert(0, str(Path(__file__).parent))
from eval_split import split_train  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--show", action="store_true", help="in các câu có Recall@k < 1")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    pipeline = HybridPipeline()
    pipeline.index(load_corpus())
    half, _ = split_train()

    scores: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row in read_jsonl(DATA_DIR / "train" / "questions.jsonl"):
        if not row["relevant_ids"] or row["id"] not in half:
            continue
        ids = pipeline.run(row["query"]).retrieved_ids
        relevant = set(row["relevant_ids"])
        s = (
            recall_at_k(ids, relevant, args.k),
            reciprocal_rank_at_k(ids, relevant, args.k),
            ndcg_at_k(ids, relevant, args.k),
        )
        for part in (half[row["id"]], "tất cả"):
            scores[part].append(s)
        if args.show and s[0] < 1:
            print(f"{row['id']} relevant={sorted(relevant)} top={ids[: args.k]}")

    print(f"{'nửa':7s} {'n':>4s} {'Recall':>7s} {'MRR':>7s} {'NDCG':>7s} {'TB':>7s}")
    for part, rows in sorted(scores.items()):
        avg = [sum(col) / len(rows) for col in zip(*rows, strict=True)]
        cells = " ".join(f"{v:7.4f}" for v in avg)
        print(f"{part:7s} {len(rows):4d} {cells} {sum(avg) / 3:7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Chấm riêng phần truy xuất của HybridPipeline (mặc định trên tập train), chia theo kiểu câu hỏi.

Không gọi bước sinh câu trả lời nên rẻ và nhanh:
    python scripts/eval_retrieval.py --offline      # chỉ BM25, không gọi API
    python scripts/eval_retrieval.py --no-rewrite   # BM25 + bge-m3 (embedding gần như miễn phí)
    python scripts/eval_retrieval.py                # thêm bước viết lại câu hỏi (tốn LLM, có cache)
    python scripts/eval_retrieval.py --questions eval/abbrev_dev.jsonl   # bộ câu viết tắt riêng
    python scripts/eval_retrieval.py --half hold --reranker BAAI/bge-reranker-v2-m3   # cross-encoder

Cột Phủ@12 là tỉ lệ đoạn đúng nằm trong `n_candidates` đoạn đầu (ứng viên đưa cho LLM).

Điểm retrieval trong công thức Benchmark = trung bình Recall@5, MRR@5, NDCG@5.
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
    parser.add_argument("--offline", action="store_true", help="chỉ BM25, không gọi gateway")
    parser.add_argument("--no-rewrite", action="store_true", help="bỏ bước viết lại câu hỏi")
    parser.add_argument("--club-weight", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--reranker", help="cross-encoder local, ví dụ BAAI/bge-reranker-v2-m3")
    parser.add_argument("--rerank-top", type=int, default=20)
    parser.add_argument(
        "--doc2query", nargs="*", default=[], choices=["bm25", "dense"], help="dùng câu hỏi giả"
    )
    parser.add_argument("--half", choices=["tune", "hold"], help="chỉ một nửa của tập train")
    parser.add_argument(
        "--questions",
        type=Path,
        default=DATA_DIR / "train" / "questions.jsonl",
        help="file JSONL có query, relevant_ids, type",
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    pipeline = HybridPipeline(
        rewrite=not (args.offline or args.no_rewrite),
        dense=not args.offline,
        club_weight=args.club_weight,
        reranker=args.reranker,
        rerank_top=args.rerank_top,
        doc2query=args.doc2query,
    )
    pipeline.index(load_corpus())
    half = split_train()[0] if args.half else {}

    by_type: dict[str, list[tuple[float, ...]]] = defaultdict(list)
    for row in read_jsonl(args.questions):
        if not row["relevant_ids"] or (args.half and half.get(row["id"]) != args.half):
            continue
        plan = pipeline.plan(row["query"])
        docs = pipeline.retrieve(plan)
        if args.reranker:
            docs = pipeline.rerank(plan, docs)
        ranked = [d.id for d in docs]
        relevant = set(row["relevant_ids"])
        scores = (
            recall_at_k(ranked[:10], relevant, args.k),
            reciprocal_rank_at_k(ranked[:10], relevant, args.k),
            ndcg_at_k(ranked[:10], relevant, args.k),
            len(relevant & set(ranked[: pipeline.n_candidates])) / len(relevant),
        )
        by_type["TẤT CẢ"].append(scores)
        by_type[row["type"]].append(scores)

    print(
        f"{'Kiểu câu hỏi':22s} {'n':>4s} {'Recall':>7s} {'MRR':>7s} {'NDCG':>7s} {'TB':>7s}"
        f" {'Phủ@12':>7s}"
    )
    for name, rows in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        avg = [sum(col) / len(rows) for col in zip(*rows, strict=True)]
        cells = " ".join(f"{v:7.3f}" for v in avg[:3])
        print(f"{name:22s} {len(rows):4d} {cells} {sum(avg[:3]) / 3:7.3f} {avg[3]:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

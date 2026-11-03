"""So điểm generation của các biến thể prompt trên hai nửa cố định của tập train.

    neorag run train --pipeline neorag.pipelines.hybrid:HybridPipeline --no-judge --out outputs/train_pred_v3.jsonl
    python scripts/eval_split.py outputs/eval_train_pred/per_query.jsonl outputs/eval_train_pred_v3/per_query.jsonl

Train được chia cân theo kiểu câu hỏi: trong mỗi kiểu, các câu xếp theo id rồi xen kẽ vào nửa
`tune` (đọc lỗi, chỉnh prompt) và nửa `hold` (chỉ để chấm). Hai câu train-0001, train-0019 từng
nằm trong ví dụ của prompt nên bị bỏ khỏi cả hai nửa.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from neorag.data import DATA_DIR, read_jsonl

EXCLUDED = {"train-0001", "train-0019"}
METRICS = ("rouge_l", "string_presence", "bleu_4")


def split_train() -> tuple[dict[str, str], dict[str, str]]:
    """id -> "tune" | "hold", và id -> kiểu câu hỏi."""
    by_type: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(DATA_DIR / "train" / "questions.jsonl"):
        if row["id"] not in EXCLUDED:
            by_type[row["type"]].append(row["id"])
    half, kind = {}, {}
    for name, ids in by_type.items():
        for i, qid in enumerate(sorted(ids)):
            half[qid] = "tune" if i % 2 == 0 else "hold"
            kind[qid] = name
    return half, kind


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("per_query", type=Path, nargs="+", help="per_query.jsonl của từng biến thể")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    half, kind = split_train()
    # (nửa, kiểu câu hỏi) -> biến thể -> metric -> danh sách điểm
    groups: dict[tuple[str, str], list[dict[str, list[float]]]] = defaultdict(
        lambda: [defaultdict(list) for _ in args.per_query]
    )
    for v, path in enumerate(args.per_query):
        for row in read_jsonl(path):
            if row["id"] not in half:
                continue
            for key in ((half[row["id"]], "TẤT CẢ"), (half[row["id"]], kind[row["id"]])):
                for m in METRICS:
                    if m in row:  # câu không có đáp án không có ROUGE-L / SP / BLEU
                        groups[key][v][m].append(row[m])

    names = [
        p.parent.name.removeprefix("eval_train_pred").strip("_") or "hiện tại"
        for p in args.per_query
    ]
    print("Mỗi ô: ROUGE-L / String Presence / BLEU-4")
    print(f"{'nửa':5s} {'kiểu câu hỏi':20s} {'n':>3s}  " + "  ".join(f"{n:>17s}" for n in names))
    for (part, name), variants in sorted(
        groups.items(), key=lambda kv: (kv[0][0] != "hold", kv[0][1] != "TẤT CẢ", kv[0][1])
    ):
        n = len(variants[0]["rouge_l"])
        cells = [
            "/".join(f"{sum(v[m]) / len(v[m]):.3f}" if v[m] else "  -  " for m in METRICS)
            for v in variants
        ]
        print(f"{part:5s} {name:20s} {n:3d}  " + "  ".join(f"{c:>17s}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

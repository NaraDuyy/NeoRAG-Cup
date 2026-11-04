"""So cặp từng câu giữa các file predictions train với một file gốc: chênh lệch trung bình ± sai số chuẩn.

    python scripts/paired.py outputs/train_pred_v7b.jsonl outputs/train_pred_x.jsonl ...

Mỗi dòng: chênh lệch của String Presence, ROUGE-L và trung bình hai chỉ số (không tốn API), kèm
sai số chuẩn (SE) của chênh lệch trung bình. |chênh| < 2·SE thì chưa phân biệt được với nhiễu.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean, stdev

from neorag.data import DATA_DIR, read_jsonl
from neorag.metrics import lexical


def scores(path: Path, gold: dict) -> dict[str, tuple[float, float]]:
    out = {}
    for p in read_jsonl(path):
        g = gold[p["id"]]
        if g["answerable"]:
            out[p["id"]] = (
                lexical.string_presence(p["answer"], g["key_facts"]),
                lexical.rouge_l(p["answer"], g["answer"]),
            )
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    gold = {r["id"]: r for r in read_jsonl(DATA_DIR / "train" / "questions.jsonl")}
    base, *others = [Path(p) for p in sys.argv[1:]]
    b = scores(base, gold)
    print(f"gốc: {base.stem}")
    for path in others:
        o = scores(path, gold)
        ids = sorted(b.keys() & o.keys())
        cells = []
        for name, f in (("SP", lambda s: s[0]), ("RL", lambda s: s[1]), ("TB", mean)):
            d = [f(o[i]) - f(b[i]) for i in ids]
            cells.append(f"{name} {mean(d):+.4f}±{stdev(d) / len(d) ** 0.5:.4f}")
        print(f"  {path.stem:28s} " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

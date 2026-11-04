"""Chấm nhanh nhiều file predictions trên tập train, chia theo nửa `tune` / `hold`.

Retrieval, String Presence, ROUGE-L tính tại chỗ. Response Relevancy chỉ tốn 1 lần gọi judge ngắn
mỗi câu (không cần ngữ cảnh), nên rẻ hơn nhiều so với chấm đầy đủ; Groundedness và Noise để dành
cho bản cuối cùng (`neorag evaluate train`). Kết quả judge được cache chung với bộ chấm chính thức.

    python scripts/quick_eval.py outputs/train_pred_v7b.jsonl outputs/train_pred_x.jsonl --rr
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from neorag.config import load_config
from neorag.data import DATA_DIR, read_jsonl
from neorag.metrics import lexical
from neorag.metrics.retrieval import ndcg_at_k, recall_at_k, reciprocal_rank_at_k

sys.path.insert(0, str(Path(__file__).parent))
from eval_split import split_train  # noqa: E402

COLUMNS = ("recall", "mrr", "ndcg", "sp", "rouge_l", "rr")


def _judge():
    from neorag.llm import LLMClient
    from neorag.metrics.judge import GatewayEmbedder, Judge

    jcfg = load_config()["judge"]
    llm = LLMClient.for_judge(cache_path=jcfg["cache_path"])
    return Judge(
        llm,
        chat_model=jcfg["chat_model"],
        embedder=GatewayEmbedder(llm, jcfg["embedding_model"]),
        relevancy_questions=jcfg["relevancy_questions"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictions", type=Path, nargs="+")
    parser.add_argument("--rr", action="store_true", help="chấm thêm Response Relevancy")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    gold = {r["id"]: r for r in read_jsonl(DATA_DIR / "train" / "questions.jsonl")}
    half, _ = split_train()
    judge = _judge() if args.rr else None

    print(f"{'file':28s} {'nửa':5s} " + " ".join(f"{c:>7s}" for c in COLUMNS) + "   ret    lex")
    for path in args.predictions:
        preds = [p for p in read_jsonl(path) if p["id"] in half and gold[p["id"]]["answerable"]]
        rr = {}
        if judge:
            with ThreadPoolExecutor(4) as pool:
                values = pool.map(
                    lambda p: judge.response_relevancy(gold[p["id"]]["query"], p["answer"]), preds
                )
                rr = dict(zip((p["id"] for p in preds), values, strict=True))
        rows: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for p in preds:
            g = gold[p["id"]]
            rel, ids = set(g["relevant_ids"]), p["retrieved_ids"]
            values = {
                "recall": recall_at_k(ids, rel, 5),
                "mrr": reciprocal_rank_at_k(ids, rel, 5),
                "ndcg": ndcg_at_k(ids, rel, 5),
                "sp": lexical.string_presence(p["answer"], g["key_facts"]),
                "rouge_l": lexical.rouge_l(p["answer"], g["answer"]),
            }
            if p["id"] in rr:
                values["rr"] = rr[p["id"]]
            for part in (half[p["id"]], "all"):
                for k, v in values.items():
                    rows[part][k].append(v)
        for part in ("tune", "hold", "all"):
            avg = {k: mean(v) for k, v in rows[part].items()}
            cells = " ".join(f"{avg[c]:7.4f}" if c in avg else "      -" for c in COLUMNS)
            ret = mean(avg[c] for c in ("recall", "mrr", "ndcg"))
            lex = mean(avg[c] for c in ("sp", "rouge_l", "rr") if c in avg)
            print(f"{path.stem[-28:]:28s} {part:5s} {cells} {ret:.4f} {lex:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

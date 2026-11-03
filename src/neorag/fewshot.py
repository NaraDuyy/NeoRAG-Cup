"""Chọn ví dụ văn phong (few-shot) cho bước sinh câu trả lời từ các câu hỏi train giống nhất.

Chỉ dùng câu hỏi train có đáp án: đáp án chuẩn của câu không có đáp án hay mở đầu bằng điều tài
liệu có ("Tài liệu chỉ cho biết ..."), kiểu mà judge Abstention không coi là từ chối.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from neorag.data import DATA_DIR, read_jsonl
from neorag.retrieval import BM25, rrf
from neorag.text import normalize


class FewShotBank:
    """Tìm câu hỏi train giống câu hỏi hiện tại bằng BM25, không gọi LLM hay embedding."""

    def __init__(self, rows: list[dict]):
        self.rows = [r for r in rows if r.get("answerable", True) and r.get("answer")]
        self.keys = [normalize(r["query"]) for r in self.rows]
        self.bm25 = BM25([r["query"] for r in self.rows])

    @classmethod
    def load(cls, path: Path = DATA_DIR / "train" / "questions.jsonl") -> FewShotBank:
        return cls(list(read_jsonl(path)))

    def select(
        self, query: str, extra: tuple[str, ...] | list[str] = (), k: int = 3, per_type: int = 2
    ) -> list[dict]:
        """`k` câu giống `query` (và các truy vấn `extra`) nhất, gộp bằng RRF.

        Bỏ câu trùng hẳn `query` (trên train đây là leave-one-out) và lấy tối đa `per_type` câu
        cùng kiểu câu hỏi để ví dụ đa dạng.
        """
        queries = [q for q in [query, *extra] if q]
        fused = rrf([list(np.argsort(-self.bm25.scores(q))) for q in queries])
        same = normalize(query)
        chosen: list[dict] = []
        types: Counter[str] = Counter()
        for i in sorted(fused, key=fused.get, reverse=True):
            row = self.rows[i]
            if self.keys[i] == same or types[row.get("type", "")] >= per_type:
                continue
            chosen.append(row)
            types[row.get("type", "")] += 1
            if len(chosen) == k:
                break
        return chosen

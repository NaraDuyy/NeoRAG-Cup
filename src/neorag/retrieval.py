"""Truy xuất lexical (BM25 trên văn bản bỏ dấu) và gộp nhiều bảng xếp hạng bằng RRF.

Câu hỏi của người dùng hay viết không dấu hoặc viết tắt ("clb mm tuyen tv khi nao"), nên BM25
so khớp trên âm tiết đã bỏ dấu, kèm bigram âm tiết để ưu tiên cụm từ ("tuyen thanh", "thanh vien").
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import numpy as np

from neorag.text import tokenize

_COMBINING = re.compile(r"[̀-ͯ]")

# Âm tiết (đã bỏ dấu) quá phổ biến, gần như không giúp phân biệt đoạn văn. Không đưa vào đây
# các âm tiết trùng với từ có nghĩa sau khi bỏ dấu, ví dụ "ban" (bạn / Ban Đối ngoại), "da" (đã / đa).
_STOPWORDS = (
    "la va cua co cac nhung duoc cho voi trong khi thi nao gi ai bao nhieu khong ko k "
    "a ah e em anh chi minh toi oi nhe vay the do nay kia ve tu den cung con "
    "mot hay hoac neu nhu de dang se rat lam"
)
STOPWORDS = frozenset(_STOPWORDS.split())


# Viết tắt hay gặp trong câu hỏi (đã bỏ dấu) -> cụm từ đầy đủ dùng trong corpus
_ABBREVIATIONS = {
    "mm": "multimedia", "mpc": "multimedia", "dpt": "da phuong tien multimedia",
    "lt": "lap trinh", "pro": "proptit", "tv": "thanh vien", "ttv": "tuyen thanh vien",
    "nk": "nhiem ky", "cn": "chu nhiem", "bcn": "ban chu nhiem", "hv": "hoc vien",
    "dk": "dang ky", "dc": "duoc", "sv": "sinh vien", "j": "gi", "v": "vay",
    "ko": "khong", "k": "khong", "ntn": "nhu the nao", "sk": "su kien",
}  # fmt: skip
# Cách gọi tên CLB không chính thức -> tên trong corpus
_CLUB_ALIASES = [
    (re.compile(r"\bclb (?:lam )?game\b|\bnha sang tao game\b"), "gcc nha sang tao game"),
    (re.compile(r"\bclb (?:da phuong tien|multimedia|mm)\b"), "multimedia"),
    (re.compile(r"\bclb (?:code|lap trinh|lt)\b"), "lap trinh proptit"),
    (re.compile(r"\bt(1[0-2]|[1-9])\b"), r"thang \1"),
]


def expand_query(query: str) -> str:
    """Thêm dạng đầy đủ của viết tắt và tên CLB không chính thức vào câu hỏi (dùng cho BM25)."""
    plain = " ".join(strip_accents(t) for t in tokenize(query))
    extra = [_ABBREVIATIONS[t] for t in plain.split() if t in _ABBREVIATIONS]
    for pattern, replacement in _CLUB_ALIASES:
        extra += [m.expand(replacement) for m in pattern.finditer(plain)]
    return " ".join([query, *extra])


def strip_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt: "Lập Trình" -> "Lap Trinh", "đ" -> "d"."""
    text = unicodedata.normalize("NFD", text)
    text = _COMBINING.sub("", text)
    return unicodedata.normalize("NFC", text).replace("đ", "d").replace("Đ", "D")


def lexical_terms(text: str) -> list[str]:
    """Âm tiết bỏ dấu (trừ stopword) và bigram âm tiết."""
    syllables = [strip_accents(t) for t in tokenize(text)]
    unigrams = [t for t in syllables if t not in STOPWORDS]
    bigrams = [f"{a}_{b}" for a, b in zip(syllables, syllables[1:], strict=False)]
    return unigrams + bigrams


class BM25:
    """Okapi BM25 trên một corpus nhỏ (vài trăm đoạn), tính toàn bộ trong bộ nhớ."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_terms = [Counter(lexical_terms(t)) for t in texts]
        self.doc_len = np.array([sum(c.values()) for c in self.doc_terms], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if len(texts) else 0.0
        df = Counter(term for c in self.doc_terms for term in c)
        n = len(texts)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        result = np.zeros(len(self.doc_terms), dtype=np.float32)
        for term in set(lexical_terms(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self.doc_terms):
                tf = counts.get(term, 0)
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                    result[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return result


def rrf(
    rankings: list[list[int]], weights: list[float] | None = None, k: int = 60
) -> dict[int, float]:
    """Reciprocal Rank Fusion: điểm của mỗi phần tử = Σ weight / (k + hạng)."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, 1):
            fused[item] = fused.get(item, 0.0) + weight / (k + rank)
    return fused

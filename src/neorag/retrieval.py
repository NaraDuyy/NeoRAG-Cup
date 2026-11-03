"""Truy xuất lexical (BM25 trên văn bản bỏ dấu) và gộp nhiều bảng xếp hạng bằng RRF.

Câu hỏi của người dùng hay viết không dấu hoặc viết tắt ("clb mm tuyen tv khi nao"), nên BM25
so khớp trên âm tiết đã bỏ dấu, kèm bigram âm tiết để ưu tiên cụm từ ("tuyen thanh", "thanh vien").
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

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
_ACRONYM_AI = re.compile(r"\bAI\b")


RESOURCE = Path(__file__).parent / "resources" / "abbreviations.tsv"


class Abbreviations:
    """Từ điển viết tắt (đã bỏ dấu) -> các nghĩa có thể, xem scripts/build_abbreviations.py.

    Không bỏ nghĩa nào: `rank_by_corpus` chỉ đưa các nghĩa xuất hiện trong corpus lên trước
    (kể cả chính token nếu nó là một từ trong corpus, ví dụ "cv" trong "vòng CV").

    Viết tắt "đã biết" (mục soạn tay, nguồn `manual`) đã được giải thích trong prompt viết lại câu hỏi:
    không sinh gợi ý cho chúng, và BM25 chỉ mở chúng theo nghĩa soạn tay đầu tiên.
    """

    def __init__(self, table: dict[str, list[str]], known: dict[str, list[str]] | None = None):
        self.table = table
        self.known = known or {}  # viết tắt đã biết -> các nghĩa soạn tay
        self.in_corpus: dict[str, list[str]] = {}

    @classmethod
    def load(cls, path: Path = RESOURCE) -> Abbreviations:
        table, known = {}, {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                token, senses, origins = line.split("\t")
                pairs = list(zip(senses.split("|"), origins.split("|"), strict=True))
                table[token] = [s for s, _ in pairs]
                if manual := [s for s, o in pairs if o == "manual"]:
                    known[token] = manual
        return cls(table, known)

    def rank_by_corpus(self, texts: list[str]) -> Abbreviations:
        haystack = " " + " ".join(strip_accents(t) for x in texts for t in tokenize(x)) + " "
        for token, senses in self.table.items():
            found = [s for s in senses if f" {strip_accents(s)} " in haystack]
            if f" {token} " in haystack and token not in senses:
                # bản thân token là một từ trong corpus (cv, gen): giữ nguyên cũng là một nghĩa
                senses, found = [token, *senses], [token, *found]
            self.in_corpus[token] = found
            self.table[token] = found + [s for s in senses if s not in found]
        return self

    def senses(self, token: str) -> list[str]:
        return self.table.get(token, [])

    def bm25_senses(self, token: str) -> list[str]:
        """Nghĩa dùng để mở rộng câu hỏi cho BM25.

        Viết tắt đã biết: nghĩa soạn tay đầu tiên. Còn lại: nghĩa có trong corpus (nếu đã xếp theo
        corpus), gộp các nghĩa trùng nhau sau khi bỏ dấu ("học viện" / "học viên").
        """
        if token in self.known:
            return self.known[token][:1]
        unique: dict[str, str] = {}
        for sense in self.in_corpus.get(token, self.senses(token)):
            unique.setdefault(strip_accents(sense), sense)
        return list(unique.values())

    def hints(self, query: str, limit: int = 3) -> list[tuple[str, list[str]]]:
        """Các viết tắt lạ (không phải viết tắt đã biết) trong câu hỏi, tối đa `limit` nghĩa mỗi cái."""
        tokens = dict.fromkeys(strip_accents(t) for t in tokenize(query))
        known = self.known
        return [(t, self.senses(t)[:limit]) for t in tokens if t in self.table and t not in known]


_DEFAULT: Abbreviations | None = None


def default_abbreviations() -> Abbreviations:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Abbreviations.load()
    return _DEFAULT


# Cách gọi tên CLB không chính thức -> tên trong corpus
_CLUB_ALIASES = [
    (re.compile(r"\bclb (?:lam )?game\b|\bnha sang tao game\b"), "gcc nha sang tao game"),
    (re.compile(r"\bclb (?:da phuong tien|multimedia|mm)\b"), "multimedia"),
    (re.compile(r"\bclb (?:code|lap trinh|lt)\b"), "lap trinh proptit"),
]


def expand_query(query: str, abbrevs: Abbreviations | None = None, ambiguous: bool = True) -> str:
    """Thêm dạng đầy đủ của viết tắt và tên CLB không chính thức vào câu hỏi (dùng cho BM25).

    `ambiguous=False` bỏ qua viết tắt có nhiều nghĩa, để bước viết lại câu hỏi bằng LLM chọn nghĩa.
    """
    abbrevs = abbrevs or default_abbreviations()
    plain = " ".join(strip_accents(t) for t in tokenize(query)).replace("cau lac bo", "clb")
    extra = []
    for token in plain.split():
        senses = abbrevs.bm25_senses(token)
        if ambiguous or len(senses) == 1:
            extra += senses
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
    # "AI" viết hoa là trí tuệ nhân tạo, không phải đại từ "ai" (stopword) sau khi về chữ thường
    syllables = [strip_accents(t) for t in tokenize(_ACRONYM_AI.sub("AI_", text))]
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

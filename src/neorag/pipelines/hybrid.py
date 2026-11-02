"""Pipeline cải tiến: viết lại câu hỏi -> hybrid search (BM25 + bge-m3, gộp RRF) -> LLM vừa chọn
đoạn liên quan vừa trả lời trong một lần gọi.

1. Viết lại câu hỏi (1 lần gọi LLM): thêm dấu, mở viết tắt, nhận diện CLB được hỏi, tách câu hỏi
   so sánh / nhiều ý thành các truy vấn con.
2. Truy xuất: với mỗi truy vấn (gốc, đã viết lại, truy vấn con) lấy xếp hạng BM25 (bỏ dấu) và
   dense (bge-m3), gộp bằng RRF, ưu tiên đoạn văn của CLB được hỏi.
3. Sinh câu trả lời (1 lần gọi LLM): đưa `n_candidates` đoạn đầu vào prompt, LLM trả về id các
   đoạn đã dùng (đưa lên đầu `retrieved_ids`) và câu trả lời ngắn, bám tài liệu.

Mỗi câu hỏi tốn 2 lần gọi `claude-haiku-5-5`; kết quả được cache trong `.cache/hybrid.sqlite`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

from neorag.data import Document
from neorag.llm import LLMClient
from neorag.pipelines.base import RAGOutput, RAGPipeline
from neorag.pipelines.baseline import GatewayEmbedder
from neorag.retrieval import BM25, expand_query, rrf, strip_accents

# Mã CLB dùng trong prompt viết lại câu hỏi -> giá trị trường `club` trong corpus
CLUBS = {
    "proptit": "CLB Lập Trình PTIT",
    "multimedia": "CLB Multimedia PTIT",
    "gcc": "CLB Nhà Sáng Tạo Game PTIT (GCC)",
    "s2b": "Cộng đồng S2B",
}

REWRITE_SYSTEM = """Bạn chuẩn hóa câu hỏi của sinh viên PTIT về cộng đồng S2B, gồm 3 CLB:
- proptit: CLB Lập Trình PTIT (PROPTIT, ProPTIT, "clb lt", "clb code", "clb lập trình của trường/học viện")
- multimedia: CLB Multimedia PTIT (MPC, "mm", "clb đa phương tiện", "clb mul")
- gcc: CLB Nhà Sáng Tạo Game PTIT (GCC, "clb game", "clb làm game")
- s2b: Cộng đồng S2B (chung cho cả 3 CLB)
Câu hỏi có thể viết không dấu, viết tắt (tv = thành viên, nk = nhiệm kỳ, cn = chủ nhiệm, hv = học viện,
dk = đăng ký, t10 = tháng 10, ko = không, dc = được, j = gì...) hoặc dùng cách gọi không chính thức.

Trả về DUY NHẤT một object JSON, không giải thích:
{"query": "<câu hỏi viết lại: tiếng Việt có dấu, mở hết viết tắt, dùng tên CLB đầy đủ, giữ nguyên ý>",
 "clubs": ["<mã các CLB câu hỏi nhắc tới; rỗng nếu không rõ>"],
 "sub_queries": ["<nếu câu hỏi có nhiều ý hoặc so sánh nhiều CLB: mỗi ý một câu hỏi ngắn độc lập; ngược lại để rỗng>"]}"""

ANSWER_SYSTEM = """Bạn là trợ lý hỏi đáp về cộng đồng S2B tại Học viện Công nghệ Bưu chính Viễn thông (PTIT):
CLB Lập Trình PTIT (ProPTIT), CLB Multimedia PTIT (MPC) và CLB Nhà Sáng Tạo Game PTIT (GCC).

Quy tắc:
1. Chỉ dùng thông tin trong TÀI LIỆU, không dùng hiểu biết bên ngoài, không suy đoán.
2. Trả lời ngắn gọn, đi thẳng vào ý: thường 1-3 câu hoàn chỉnh, nhắc lại chủ thể (tên CLB, sự kiện).
   Không chào hỏi, không mời hỏi thêm, không dùng markdown, không ghi số hiệu tài liệu.
3. Giữ nguyên tên riêng, ngày tháng, con số, tên vòng/sự kiện, đường link đúng như trong tài liệu.
4. Thông tin gắn với một thời điểm (đợt tuyển, nhiệm kỳ, năm) thì nêu rõ mốc thời gian đó.
5. Nếu tài liệu không có thông tin để trả lời: nói rõ "Tài liệu hiện có không có thông tin về ..." và
   không đưa ra con số, tên hay quy định nào như thể đó là câu trả lời. Có thể nói thêm ngắn gọn
   thông tin liên quan mà tài liệu có (ví dụ thông tin đó thuộc CLB khác hoặc năm khác).
6. Nếu tài liệu chỉ trả lời được một phần, trả lời phần có và nói rõ phần nào tài liệu không nêu.
7. Nếu các nguồn ghi khác nhau, nêu cả hai và nguồn của từng thông tin.

Định dạng trả lời, đúng 2 phần:
ID: <id các tài liệu đã dùng để trả lời, xếp từ quan trọng nhất, cách nhau bởi dấu phẩy; để trống nếu không dùng tài liệu nào>
TRẢ LỜI: <câu trả lời>

Ví dụ về văn phong câu trả lời:
- Hỏi: "CLB Lập Trình PTIT được thành lập ngày nào và phương châm hoạt động là gì?"
  TRẢ LỜI: CLB Lập Trình PTIT (PROPTIT) được thành lập ngày 09/10/2011, với phương châm hoạt động "Chia sẻ để cùng nhau phát triển".
- Hỏi: "Phí hoặc quỹ hằng tháng khi tham gia CLB Multimedia PTIT là bao nhiêu?"
  TRẢ LỜI: Tài liệu hiện có không có thông tin về phí hoặc quỹ của CLB Multimedia PTIT, nên mình không thể trả lời chính xác câu hỏi này."""

ANSWER_USER = """TÀI LIỆU:
{context}

CÂU HỎI: {query}{rewritten}"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_ID_LINE = re.compile(r"^\W*ID\W*:(.*)$", re.MULTILINE | re.IGNORECASE)
_ANSWER_START = re.compile(r"^\W*TRẢ LỜI\W*:\s*", re.MULTILINE | re.IGNORECASE)


@dataclass
class QueryPlan:
    """Kết quả phân tích câu hỏi: các truy vấn dùng để tìm kiếm và các CLB được hỏi."""

    query: str
    rewritten: str = ""
    clubs: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)

    @property
    def search_queries(self) -> list[str]:
        queries = [self.query, self.rewritten, *self.sub_queries]
        return list(dict.fromkeys(q for q in queries if q))


def detect_clubs(query: str) -> list[str]:
    """Nhận diện CLB được nhắc tới bằng từ khóa (dùng khi không viết lại câu hỏi bằng LLM)."""
    text = strip_accents(expand_query(query)).lower()
    keywords = {
        "proptit": ("proptit", "lap trinh"),
        "multimedia": ("multimedia", "da phuong tien"),
        "gcc": ("gcc", "nha sang tao game"),
        "s2b": ("s2b",),
    }
    return [code for code, words in keywords.items() if any(w in text for w in words)]


def parse_plan(query: str, reply: str) -> QueryPlan:
    """Đọc JSON của bước viết lại câu hỏi; reply hỏng thì chỉ dùng câu hỏi gốc."""
    try:
        data = json.loads(_JSON_OBJECT.search(reply).group())
    except (AttributeError, ValueError):
        return QueryPlan(query, clubs=detect_clubs(query))
    rewritten = data.get("query") if isinstance(data.get("query"), str) else ""
    clubs = [c for c in data.get("clubs") or [] if c in CLUBS]
    subs = [s for s in data.get("sub_queries") or [] if isinstance(s, str) and s.strip()]
    return QueryPlan(query, rewritten.strip(), clubs, subs[:4])


def parse_answer(reply: str, allowed_ids: set[int]) -> tuple[str, list[int]]:
    """Tách phần `ID:` và `TRẢ LỜI:` của câu trả lời. Trả về (answer, id đã dùng)."""
    ids: list[int] = []
    if m := _ID_LINE.search(reply):
        ids = [int(x) for x in re.findall(r"\d+", m.group(1)) if int(x) in allowed_ids]
    start = _ANSWER_START.search(reply)
    answer = reply[start.end() :] if start else _ID_LINE.sub("", reply)
    answer = re.sub(r"\*\*|__", "", answer).strip()
    return answer, list(dict.fromkeys(ids))


class HybridPipeline(RAGPipeline):
    def __init__(
        self,
        n_candidates: int = 12,
        n_retrieved: int = 10,
        rewrite: bool = True,
        dense: bool = True,
        club_weight: float = 0.5,
        llm_model: str | None = None,
        embedding_model: str | None = None,
        llm: LLMClient | None = None,
        embedder=None,
    ):
        self.n_candidates = n_candidates  # số đoạn đưa vào prompt sinh câu trả lời
        self.n_retrieved = n_retrieved  # số đoạn trả về để chấm retrieval
        self.rewrite = rewrite
        self.dense = dense
        self.club_weight = club_weight  # trọng số RRF của bảng xếp hạng chỉ gồm CLB được hỏi
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "claude-haiku-5-5")
        self.llm = llm or LLMClient(cache_path=".cache/hybrid.sqlite")
        if dense:
            self.embedder = embedder or GatewayEmbedder(
                self.llm, embedding_model or os.getenv("EMBEDDING_MODEL", "bge-m3")
            )

    def index(self, corpus: list[Document]) -> None:
        self.docs = corpus
        self.by_id = {d.id: d for d in corpus}
        self.bm25 = BM25([f"{d.club} {d.topic} {d.text}" for d in corpus])
        if self.dense:
            self.doc_vectors = self.embedder.embed([d.text for d in corpus])

    def _chat(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.llm.chat(messages, model=self.llm_model)

    def plan(self, query: str) -> QueryPlan:
        if not self.rewrite:
            return QueryPlan(query, clubs=detect_clubs(query))
        return parse_plan(query, self._chat(REWRITE_SYSTEM, query))

    def retrieve(self, plan: QueryPlan) -> list[Document]:
        """Toàn bộ corpus, xếp theo điểm RRF giảm dần."""
        queries = plan.search_queries
        rankings, weights = [], []
        for q in queries:
            rankings.append(list(np.argsort(-self.bm25.scores(expand_query(q)))))
            weights.append(1.0)
        if self.dense:
            q_vectors = self.embedder.embed(queries)
            for vec in q_vectors:
                rankings.append(list(np.argsort(-(self.doc_vectors @ vec))))
                weights.append(1.0)
        clubs = {CLUBS[c] for c in plan.clubs} | {CLUBS["s2b"]}
        if plan.clubs and len(clubs) < len(CLUBS):
            # thêm một bảng xếp hạng chỉ gồm đoạn văn của CLB được hỏi để đẩy chúng lên
            in_club = {i for i, d in enumerate(self.docs) if d.club in clubs}
            for ranking in list(rankings):
                rankings.append([i for i in ranking if i in in_club])
                weights.append(self.club_weight)
        fused = rrf(rankings, weights)
        return [self.docs[i] for i in sorted(fused, key=fused.get, reverse=True)]

    def answer(self, plan: QueryPlan, candidates: list[Document]) -> tuple[str, list[int]]:
        context = "\n\n".join(f"[{d.id}] ({d.club} · {d.topic}) {d.text}" for d in candidates)
        rewritten = ""
        if plan.rewritten and plan.rewritten.lower() != plan.query.lower():
            rewritten = f"\n(Câu hỏi đã chuẩn hóa: {plan.rewritten})"
        user = ANSWER_USER.format(context=context, query=plan.query, rewritten=rewritten)
        reply = self._chat(ANSWER_SYSTEM, user)
        return parse_answer(reply, {d.id for d in candidates})

    def run(self, query: str) -> RAGOutput:
        plan = self.plan(query)
        ranked = self.retrieve(plan)
        answer, used = self.answer(plan, ranked[: self.n_candidates])
        if not answer:
            answer = "Tài liệu hiện có không có thông tin để trả lời câu hỏi này."
        retrieved = list(dict.fromkeys([*used, *(d.id for d in ranked)]))[: self.n_retrieved]
        return RAGOutput(answer=answer, retrieved_ids=retrieved)

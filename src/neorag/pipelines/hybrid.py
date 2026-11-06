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
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from neorag.data import Document
from neorag.fewshot import FewShotBank
from neorag.llm import LLMClient
from neorag.metrics.lexical import rouge_l
from neorag.pipelines.base import RAGOutput, RAGPipeline
from neorag.pipelines.baseline import GatewayEmbedder
from neorag.retrieval import (
    BM25,
    Abbreviations,
    expand_query,
    load_doc2query,
    rrf,
    strip_accents,
)
from neorag.text import split_sentences

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

ANSWER_RULES = """Bạn là trợ lý hỏi đáp về cộng đồng S2B tại Học viện Công nghệ Bưu chính Viễn thông (PTIT):
CLB Lập Trình PTIT (ProPTIT), CLB Multimedia PTIT (MPC) và CLB Nhà Sáng Tạo Game PTIT (GCC).

Quy tắc:
1. Chỉ dùng thông tin trong TÀI LIỆU, không dùng hiểu biết bên ngoài, không suy đoán.
2. Trả lời ngắn gọn, đi thẳng vào đúng điều được hỏi: thường 1-2 câu hoàn chỉnh, tối đa 3 câu,
   nhắc lại chủ thể (tên CLB, sự kiện). Không chào hỏi, không mời hỏi thêm, không dùng markdown,
   không ghi số hiệu tài liệu (như "[55]"), không mở đầu bằng cụm chung chung "Theo tài liệu".
3. Nêu đủ mọi chi tiết cụ thể trả lời trực tiếp câu hỏi (con số, thời hạn, tên, điều kiện), giữ nguyên
   tên riêng, ngày tháng, con số, tên vòng/sự kiện, đường link đúng như trong tài liệu. Với ý chính,
   dùng lại đúng cụm từ của tài liệu thay vì diễn đạt lại.
4. Thông tin gắn với một thời điểm (đợt tuyển, nhiệm kỳ, năm) thì nêu rõ mốc thời gian đó.
5. Câu hỏi có/không hoặc hỏi để xác nhận: câu trả lời mở đầu bằng từ kết luận đứng riêng, kết thúc
   bằng dấu chấm ("Có." / "Không." / "Đúng." / "Chưa chắc."), rồi mới nêu chi tiết.
6. Câu hỏi so sánh hoặc hỏi về nhiều đối tượng: nêu từng đối tượng theo đúng thứ tự câu hỏi nhắc tới,
   giữ nguyên tên gọi.
7. Đã trả lời được câu hỏi thì dừng: KHÔNG thêm câu kiểu "Tài liệu không nêu thêm ..." hay bình luận
   về những chi tiết người hỏi không hỏi. Không tự suy đoán ngoài tài liệu (như "có thể liên quan
   đến ...", "nhiều khả năng ...").
8. Nếu tài liệu không có thông tin để trả lời: câu trả lời PHẢI mở đầu bằng "Tài liệu hiện có không có
   thông tin về ...", rồi có thể thêm một câu ngắn "Tài liệu chỉ cho biết ..." về điều liên quan mà
   tài liệu có (ví dụ thông tin đó thuộc CLB khác hoặc năm khác). Không đưa ra con số, tên hay quy
   định nào như thể đó là câu trả lời.
9. Nếu câu hỏi có nhiều ý mà tài liệu chỉ có một phần: trả lời phần có, rồi nói ngắn gọn ý nào tài
   liệu không nêu ("tài liệu không có thông tin về ...").
10. Nếu các nguồn ghi khác nhau, nêu cả hai và nguồn của từng thông tin."""

ANSWER_FORMAT = """Định dạng trả lời, đúng 2 phần:
ID: <id các tài liệu có thông tin liên quan trực tiếp tới câu hỏi, cách nhau bởi dấu phẩy: tài liệu dùng để trả lời xếp trước, rồi tới tài liệu tổng quan hoặc bổ trợ cùng nội dung dù không trích; không ghi tài liệu chỉ trùng từ khóa mà nói về chuyện khác; để trống nếu không có tài liệu nào liên quan>
TRẢ LỜI: <câu trả lời>"""

EXAMPLE_ANSWER = """- Hỏi: "CLB Lập Trình PTIT có hoạt động thể thao nào cho thành viên không?"
  TRẢ LỜI: Có. CLB Lập Trình PTIT tổ chức PROPTIT Marathon, cuộc thi chạy của thành viên hưởng ứng tinh thần "Có sức khỏe là có tất cả", và tổ chức thêm Hội thao cho thành viên."""

EXAMPLE_ABSTAIN = """- Hỏi: "Game "Đế Vương Phải Có Long Ngai" của GCC có bản cho điện thoại không?"
  TRẢ LỜI: Tài liệu hiện có không có thông tin về bản cho điện thoại của trò chơi "Đế Vương Phải Có Long Ngai". Tài liệu chỉ cho biết trang itch.io của CLB Nhà Sáng Tạo Game PTIT (GCC) đăng trò chơi này (thể loại Action) với bản tải về cho Windows."""

ANSWER_SYSTEM = f"""{ANSWER_RULES}

{ANSWER_FORMAT}

Ví dụ về văn phong câu trả lời:
{EXAMPLE_ANSWER}
{EXAMPLE_ABSTAIN}"""

# Có ví dụ động (few-shot) thì chỉ giữ ví dụ tĩnh cho câu không có đáp án
ANSWER_SYSTEM_FEWSHOT = f"""{ANSWER_RULES}
11. Phần VÍ DỤ VĂN PHONG trong tin nhắn là các câu hỏi khác, chỉ để học cách viết (thứ tự ý, cách
    mở đầu, độ dài); mọi thông tin trong câu trả lời chỉ được lấy từ TÀI LIỆU.

{ANSWER_FORMAT}

Ví dụ câu trả lời khi tài liệu không có thông tin:
{EXAMPLE_ABSTAIN}"""

REFUSAL_PREFIX = "Tài liệu hiện có không có thông tin"

HINTS_HEADER = "Gợi ý viết tắt (có thể sai, hãy chọn theo ngữ cảnh hoặc bỏ qua):"

EXAMPLES_HEADER = (
    "VÍ DỤ VĂN PHONG (câu hỏi khác, chỉ để tham khảo cách viết; KHÔNG dùng làm nguồn thông tin):"
)

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


def mbr_select(
    samples: list[tuple[str, list[int]]], ids_mode: str = "chosen"
) -> tuple[str, list[int]]:
    """Minimum Bayes Risk: chọn câu trả lời có ROUGE-L trung bình cao nhất so với các mẫu còn lại.

    Câu lệch hẳn (thiếu ý, lan man, hiểu sai) ít giống các mẫu khác nên bị loại; nhờ vậy giảm
    phương sai do sinh ngẫu nhiên. `ids_mode="vote"` lấy các id được quá nửa số mẫu ghi, xếp theo
    số phiếu rồi vị trí trung bình (không có id nào đạt đa số thì dùng id của câu được chọn).
    """
    if len(samples) == 1:
        return samples[0]
    answers = [a for a, _ in samples]
    utility = [
        sum(rouge_l(a, b) for j, b in enumerate(answers) if j != i) for i, a in enumerate(answers)
    ]
    best = max(range(len(samples)), key=lambda i: (utility[i], -i))
    answer, ids = samples[best]
    if ids_mode == "vote":
        votes: Counter[int] = Counter()
        positions: dict[int, list[int]] = {}
        for _, sample_ids in samples:
            for pos, i in enumerate(sample_ids):
                votes[i] += 1
                positions.setdefault(i, []).append(pos)
        majority = [i for i in votes if 2 * votes[i] > len(samples)]
        majority.sort(key=lambda i: (-votes[i], sum(positions[i]) / len(positions[i])))
        ids = majority or ids
    return answer, ids


class HybridPipeline(RAGPipeline):
    def __init__(
        self,
        n_candidates: int = 30,
        n_retrieved: int = 10,
        n_promoted: int = 5,
        few_shot: int = 3,
        mbr: int = 1,
        mbr_ids: str = "chosen",
        trim_refusal: bool = True,
        evidence_weight: float = 0.0,
        rewrite: bool = True,
        dense: bool = True,
        club_weight: float = 0.5,
        llm_model: str | None = None,
        embedding_model: str | None = None,
        llm: LLMClient | None = None,
        embedder=None,
        reranker="BAAI/bge-reranker-v2-m3",
        rerank_top: int = 30,
        rerank_weight: float = 2.0,
        doc2query: tuple[str, ...] = (),
        rerank_candidates: bool = False,
        rerank_tail: bool = True,
    ):
        self.n_candidates = n_candidates  # số đoạn đưa vào prompt sinh câu trả lời
        self.n_retrieved = n_retrieved  # số đoạn trả về để chấm retrieval
        self.n_promoted = n_promoted  # số id ở dòng `ID:` tối đa được đưa lên đầu
        self.few_shot = few_shot  # số câu train giống nhất làm ví dụ văn phong; 0 là tắt
        self.mbr = mbr  # số câu trả lời sinh ra để chọn câu đồng thuận nhất (MBR); 1 là tắt
        self.mbr_ids = mbr_ids  # "chosen": id của câu được chọn; "vote": id được đa số mẫu ghi
        # câu từ chối chỉ giữ câu đầu: thông tin kèm theo (nhất là phí, con số) đôi khi khiến judge
        # Abstention coi là đã trả lời, mà câu không có đáp án thì chỉ được chấm Abstention
        self.trim_refusal = trim_refusal
        # trọng số RRF của thứ hạng theo độ giống câu trả lời khi xếp phần đuôi; 0 là tắt
        self.evidence_weight = evidence_weight
        # cross-encoder chạy local (tên model hoặc object có .score); None là tắt
        if isinstance(reranker, str):
            from neorag.embeddings import LocalReranker

            reranker = LocalReranker(reranker)
        self.reranker = reranker
        self.rerank_top = rerank_top  # số đoạn đầu RRF được cross-encoder chấm lại
        self.rerank_weight = rerank_weight  # trọng số RRF của thứ tự cross-encoder
        # nơi dùng câu hỏi giả của doc2query: "bm25" (nối vào văn bản), "dense" (bảng xếp hạng riêng)
        self.doc2query = tuple(doc2query)
        self.rerank_candidates = rerank_candidates  # chọn ứng viên cho LLM theo cross-encoder
        self.rerank_tail = rerank_tail  # xếp phần sau các id LLM ghi theo cross-encoder
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
        texts = [f"{d.club} {d.topic} {d.text}" for d in corpus]
        # doc2query: câu hỏi giả của từng đoạn (scripts/build_doc2query.py)
        questions = load_doc2query() if self.doc2query else {}
        doc_questions = [questions.get(d.id, []) for d in corpus]
        if "bm25" in self.doc2query:
            self.bm25 = BM25(
                [" ".join([t, *qs]) for t, qs in zip(texts, doc_questions, strict=True)]
            )
        else:
            self.bm25 = BM25(texts)
        # nghĩa viết tắt có trong corpus được xếp trước, nhưng không nghĩa nào bị bỏ
        self.abbrevs = Abbreviations.load().rank_by_corpus(texts)
        if self.few_shot:
            self.examples = FewShotBank.load()
        if self.dense:
            self.doc_vectors = self.embedder.embed([d.text for d in corpus])
            if "dense" in self.doc2query:
                flat = [(i, q) for i, qs in enumerate(doc_questions) for q in qs]
                self.question_doc = np.array([i for i, _ in flat])
                self.question_vectors = self.embedder.embed([q for _, q in flat])

    def _chat(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.llm.chat(messages, model=self.llm_model)

    def plan(self, query: str) -> QueryPlan:
        if not self.rewrite:
            return QueryPlan(query, clubs=detect_clubs(query))
        return parse_plan(query, self._chat(REWRITE_SYSTEM, self._rewrite_user(query)))

    def _rewrite_user(self, query: str) -> str:
        """Câu hỏi kèm các nghĩa ứng viên của viết tắt; không có viết tắt thì giữ nguyên câu hỏi."""
        lines = [
            f"- {token}: " + " | ".join(f"{s} (giữ nguyên)" if s == token else s for s in senses)
            for token, senses in self.abbrevs.hints(query)
        ]
        return "\n".join([query, "", HINTS_HEADER, *lines]) if lines else query

    def retrieve(self, plan: QueryPlan) -> list[Document]:
        """Toàn bộ corpus, xếp theo điểm RRF giảm dần."""
        queries = plan.search_queries
        rankings, weights = [], []
        for q in queries:
            # có bước viết lại thì để LLM chọn nghĩa của viết tắt đa nghĩa, BM25 chỉ mở viết tắt một nghĩa
            expanded = expand_query(q, self.abbrevs, ambiguous=not self.rewrite)
            rankings.append(list(np.argsort(-self.bm25.scores(expanded))))
            weights.append(1.0)
        # bge-m3 hiểu kém câu viết không dấu, nên chỉ embed các truy vấn có dấu
        accented = [q for q in queries if strip_accents(q) != q]
        if self.dense and accented:
            for vec in self.embedder.embed(accented):
                rankings.append(list(np.argsort(-(self.doc_vectors @ vec))))
                weights.append(1.0)
                if "dense" in self.doc2query:
                    # điểm của đoạn = câu hỏi giả giống câu hỏi nhất
                    best = np.full(len(self.docs), -1.0)
                    np.maximum.at(best, self.question_doc, self.question_vectors @ vec)
                    rankings.append(list(np.argsort(-best)))
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
        system = ANSWER_SYSTEM
        if self.few_shot:
            examples = self.examples.select(plan.query, [plan.rewritten], k=self.few_shot)
            lines = [f'- Hỏi: "{e["query"]}"\n  TRẢ LỜI: {e["answer"]}' for e in examples]
            user = "\n".join([EXAMPLES_HEADER, *lines, "", user])
            system = ANSWER_SYSTEM_FEWSHOT
        allowed = {d.id for d in candidates}
        # mẫu thứ i thêm i dòng trống cuối tin nhắn: cùng nội dung nhưng không trúng cache của mẫu khác
        samples = [
            parse_answer(self._chat(system, user + "\n" * i), allowed) for i in range(self.mbr)
        ]
        return mbr_select(samples, self.mbr_ids)

    def rerank(self, plan: QueryPlan, ranked: list[Document]) -> list[Document]:
        """Gộp thứ tự RRF với thứ tự cross-encoder của `rerank_top` đoạn đầu (lại bằng RRF).

        Xếp hẳn theo cross-encoder kém hơn RRF trên nửa hold của train; gộp với trọng số
        `rerank_weight` thì tăng trên cả tune, hold và bộ dev viết tắt.
        """
        head = ranked[: self.rerank_top]
        texts = [f"{d.club} · {d.topic}. {d.text}" for d in head]
        scores = self.reranker.score(plan.rewritten or plan.query, texts)
        by_score = sorted(range(len(head)), key=lambda i: -scores[i])
        fused = rrf([list(range(len(ranked))), by_score], [1.0, self.rerank_weight])
        return [ranked[i] for i in sorted(fused, key=fused.get, reverse=True)]

    def evidence(self, tail: list[Document], answer: str) -> list[Document]:
        """Gộp phần đuôi với thứ hạng theo độ giống câu trả lời (BM25 + bge-m3).

        `relevant_ids` của BTC gồm mọi đoạn chứa thông tin có trong đáp án, kể cả đoạn câu hỏi
        không nhắc tới (train-0099: đoạn nội quy cũng ghi "quay lại giúp đỡ CLB"). Tìm theo câu
        trả lời bắt được những đoạn này mà tìm theo câu hỏi thì không.
        """
        position = {d.id: i for i, d in enumerate(self.docs)}
        rankings = [[position[d.id] for d in tail], list(np.argsort(-self.bm25.scores(answer)))]
        weights = [1.0, self.evidence_weight]
        if self.dense:
            vec = self.embedder.embed([answer])[0]
            rankings.append(list(np.argsort(-(self.doc_vectors @ vec))))
            weights.append(self.evidence_weight)
        fused = rrf(rankings, weights)
        return [self.docs[i] for i in sorted(fused, key=fused.get, reverse=True)]

    def run(self, query: str) -> RAGOutput:
        plan = self.plan(query)
        ranked = self.retrieve(plan)
        reranked = self.rerank(plan, ranked) if self.reranker else ranked
        pool = reranked if self.rerank_candidates else ranked
        answer, used = self.answer(plan, pool[: self.n_candidates])
        if not answer:
            answer = f"{REFUSAL_PREFIX} để trả lời câu hỏi này."
        elif self.trim_refusal and answer.startswith(REFUSAL_PREFIX):
            answer = split_sentences(answer)[0]
        tail = reranked if self.rerank_tail else ranked
        if self.evidence_weight and not answer.startswith(REFUSAL_PREFIX):
            tail = self.evidence(tail, answer)
        promoted = used[: self.n_promoted]
        retrieved = list(dict.fromkeys([*promoted, *(d.id for d in tail)]))[: self.n_retrieved]
        return RAGOutput(answer=answer, retrieved_ids=retrieved)

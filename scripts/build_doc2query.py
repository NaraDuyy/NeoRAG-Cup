"""Doc2query: sinh câu hỏi giả cho từng đoạn của corpus, dùng để mở rộng index của HybridPipeline.

Mỗi đoạn được đưa cho claude-haiku-5-5 (qua gateway, có cache) để viết vài câu hỏi mà đoạn đó trả
lời trực tiếp, với nhiều cách diễn đạt (trang trọng, đời thường, từ đồng nghĩa, tiếng Anh/Việt).
Chỉ dùng nội dung corpus, không nhìn câu hỏi train hay test.

    python scripts/build_doc2query.py            # ghi src/neorag/resources/doc2query.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from neorag.data import load_corpus
from neorag.llm import LLMClient
from neorag.retrieval import DOC2QUERY_RESOURCE

SYSTEM = """Bạn giúp xây chỉ mục tìm kiếm cho tài liệu về cộng đồng S2B tại PTIT (CLB Lập Trình PTIT,
CLB Multimedia PTIT, CLB Nhà Sáng Tạo Game PTIT). Cho một đoạn tài liệu, hãy viết {n} câu hỏi khác
nhau mà sinh viên có thể hỏi và đoạn này trả lời trực tiếp.
- Mỗi câu hỏi về một ý khác nhau của đoạn; phủ cả các chi tiết nhỏ (tên, con số, mốc thời gian).
- Đa dạng cách diễn đạt: có câu trang trọng, có câu đời thường của sinh viên, dùng từ đồng nghĩa
  hoặc cách gọi khác với từ trong đoạn (ví dụ "Medical" -> "y tế", "Chủ nhiệm" -> "trưởng CLB").
- Viết tiếng Việt có dấu, nêu tên CLB khi cần.
Mỗi câu hỏi trên một dòng, không đánh số, không giải thích."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=5, help="số câu hỏi cho mỗi đoạn")
    parser.add_argument("--model", default="claude-haiku-5-5")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    llm = LLMClient(cache_path=".cache/hybrid.sqlite")
    system = SYSTEM.format(n=args.n)

    def ask(doc):
        user = f"({doc.club} · {doc.topic})\n{doc.text}"
        reply = llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=args.model,
        )
        questions = [q.strip(" -•*\t") for q in reply.splitlines()]
        return {"id": doc.id, "questions": [q for q in questions if q.endswith("?")][: args.n]}

    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(ask, load_corpus()))
    with open(DOC2QUERY_RESOURCE, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Đã ghi {sum(len(r['questions']) for r in rows)} câu hỏi cho {len(rows)} đoạn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

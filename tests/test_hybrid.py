import json

from conftest import FakeEmbedder, FakeLLM
from neorag.data import Query
from neorag.pipelines.hybrid import (
    HINTS_HEADER,
    REWRITE_SYSTEM,
    HybridPipeline,
    detect_clubs,
    parse_answer,
    parse_plan,
)
from neorag.predict import predict
from neorag.retrieval import BM25, Abbreviations, expand_query, lexical_terms, rrf, strip_accents


def test_strip_accents_and_expand_query():
    assert strip_accents("Đội ngũ Lập Trình") == "Doi ngu Lap Trinh"
    expanded = expand_query("clb mm tuyen tv t10 ko")
    assert "multimedia" in expanded and "thành viên" in expanded and "tháng 10" in expanded
    assert "gcc" in expand_query("clb lam game cua hv")
    assert "gcc" in expand_query("câu lạc bộ game ptit")
    # viết tắt lạ đa nghĩa: chỉ mở khi ambiguous=True
    assert "bao nhiêu" in expand_query("phi bn")
    assert "bao nhiêu" not in expand_query("phi bn", ambiguous=False)
    # viết tắt đã biết: luôn mở theo nghĩa soạn tay đầu tiên, như bản trước khi có từ điển
    expanded = expand_query("hv dk t5 sv", ambiguous=False)
    assert expanded == "hv dk t5 sv học viện đăng ký tháng 5 sinh viên"


def test_known_abbreviations_get_no_hint():
    abbrevs = Abbreviations.load()
    assert abbrevs.hints("clb mm tuyen tv bh, phi bn") == [
        ("bh", abbrevs.senses("bh")[:3]),
        ("bn", abbrevs.senses("bn")[:3]),
    ]
    assert abbrevs.hints("hv t10 ko dc j") == []


def test_bm25_senses_merge_spellings_without_accents():
    abbrevs = Abbreviations({"hvien": ["học viện", "học viên"]})
    assert abbrevs.bm25_senses("hvien") == ["học viện"]


def test_abbreviations_rank_by_corpus_keeps_every_sense():
    abbrevs = Abbreviations({"t5": ["tháng 5", "thứ 5"], "cv": ["công việc"]})
    abbrevs.rank_by_corpus(["Lịch sinh hoạt vào thứ 5 hằng tuần", "Vòng CV online"])
    assert abbrevs.senses("t5") == ["thứ 5", "tháng 5"]
    assert abbrevs.bm25_senses("t5") == ["thứ 5"]
    # "cv" có trong corpus nên giữ nguyên cũng là một nghĩa
    assert abbrevs.senses("cv") == ["cv", "công việc"]
    assert abbrevs.hints("clb sinh hoat T5 ko") == [("t5", ["thứ 5", "tháng 5"])]
    assert expand_query("t5", abbrevs, ambiguous=False) == "t5 thứ 5"


def test_default_abbreviations_are_packaged():
    abbrevs = Abbreviations.load()
    assert len(abbrevs.table) > 100
    assert {"tháng 5", "thứ 5"} <= set(abbrevs.senses("t5"))
    every_sense = {s for senses in abbrevs.table.values() for s in senses}
    assert not {"súc vật", "bằng được"} & every_sense  # nghĩa rác đã bị lọc


def test_bm25_matches_query_without_diacritics(corpus):
    bm25 = BM25([d.text for d in corpus])
    assert int(bm25.scores("clb multimedia thanh lap ngay nao").argmax()) == 1
    assert bm25.scores("xyz").sum() == 0


def test_uppercase_ai_is_a_keyword_not_a_stopword():
    assert "ai_" in lexical_terms("cuộc thi AI nào")
    assert "ai" not in lexical_terms("ai là chủ nhiệm") and "ai_" not in lexical_terms("ai làm")
    bm25 = BM25(["Talkshow về xu hướng phát triển AI", "Hội thao cho thành viên"])
    assert int(bm25.scores("talkshow ve AI").argmax()) == 0
    assert bm25.scores("ai tổ chức").sum() == 0


def test_rrf_prefers_items_ranked_high_everywhere():
    fused = rrf([[1, 2, 3], [2, 1, 3]], weights=[1.0, 2.0])
    assert max(fused, key=fused.get) == 2


def test_detect_clubs():
    assert detect_clubs("clb lam game co phong rieng ko") == ["gcc"]
    assert detect_clubs("ProPTIT và MPC khác gì nhau") == ["proptit", "multimedia"]
    assert detect_clubs("Học lập trình có cần giỏi toán không") == ["proptit"]


def test_parse_plan_reads_json_and_falls_back():
    reply = 'Đây: {"query": "CLB Multimedia PTIT tuyển khi nào?", "clubs": ["multimedia", "abc"], "sub_queries": []}'
    plan = parse_plan("clb mm tuyen khi nao", reply)
    assert plan.rewritten == "CLB Multimedia PTIT tuyển khi nào?"
    assert plan.clubs == ["multimedia"]
    assert plan.search_queries == ["clb mm tuyen khi nao", "CLB Multimedia PTIT tuyển khi nào?"]

    broken = parse_plan("clb game o dau", "không phải JSON")
    assert broken.search_queries == ["clb game o dau"] and broken.clubs == ["gcc"]


def test_parse_answer():
    answer, ids = parse_answer("ID: 3, 1, 99, 3\nTRẢ LỜI: **GCC** ra đời năm 2023.", {1, 2, 3})
    assert (answer, ids) == ("GCC ra đời năm 2023.", [3, 1])
    assert parse_answer("ID:\nTRẢ LỜI: Tài liệu không có.", {1}) == ("Tài liệu không có.", [])
    assert parse_answer("Chỉ có câu trả lời.", {1}) == ("Chỉ có câu trả lời.", [])


def _responder(messages):
    if messages[0]["content"] == REWRITE_SYSTEM:
        return json.dumps(
            {"query": "CLB Multimedia PTIT thành lập ngày nào?", "clubs": ["multimedia"]},
            ensure_ascii=False,
        )
    return "ID: 2\nTRẢ LỜI: CLB Multimedia PTIT được thành lập ngày 10/11/2012."


def test_hybrid_runs_end_to_end(corpus):
    llm = FakeLLM(_responder)
    pipeline = HybridPipeline(llm=llm, embedder=FakeEmbedder())
    pipeline.prepare()
    pipeline.index(corpus)
    preds = predict(pipeline, [Query("q1", "clb mm thanh lap ngay nao")], max_retrieved=10)
    assert preds[0].answer == "CLB Multimedia PTIT được thành lập ngày 10/11/2012."
    assert preds[0].retrieved_ids[0] == 2
    assert sorted(preds[0].retrieved_ids) == [1, 2, 3]
    assert len(llm.calls) == 2  # viết lại câu hỏi + trả lời
    assert "[2] (CLB Multimedia PTIT" in llm.calls[1][1]["content"]


def test_plan_adds_abbreviation_hints_to_user_message(corpus):
    llm = FakeLLM(_responder)
    pipeline = HybridPipeline(llm=llm, embedder=FakeEmbedder())
    pipeline.index(corpus)
    pipeline.plan("bh clb mm co sinh hoat ko")
    user = llm.calls[0][1]["content"]
    assert user.startswith("bh clb mm co sinh hoat ko\n\n" + HINTS_HEADER)
    assert "- bh: " in user and "bao giờ" in user
    assert "- mm" not in user and "- ko" not in user  # viết tắt đã biết không cần gợi ý
    # không có viết tắt lạ: user message chỉ là câu hỏi, giống hệt bản trước (trúng cache cũ)
    pipeline.plan("t10 clb mm co sinh hoat ko")
    assert llm.calls[1][1]["content"] == "t10 clb mm co sinh hoat ko"
    assert llm.calls[0][0]["content"] == REWRITE_SYSTEM


def test_hybrid_without_rewrite_or_dense(corpus):
    llm = FakeLLM(lambda m: "")  # LLM trả rỗng: vẫn phải có câu trả lời hợp lệ
    pipeline = HybridPipeline(rewrite=False, dense=False, llm=llm)
    pipeline.index(corpus)
    out = pipeline.run("GCC ra doi nam nao")
    assert out.answer and out.retrieved_ids[0] == 3
    assert len(llm.calls) == 1

import json

from conftest import FakeEmbedder, FakeLLM
from neorag.data import Query
from neorag.pipelines.hybrid import (
    REWRITE_SYSTEM,
    HybridPipeline,
    detect_clubs,
    parse_answer,
    parse_plan,
)
from neorag.predict import predict
from neorag.retrieval import BM25, expand_query, rrf, strip_accents


def test_strip_accents_and_expand_query():
    assert strip_accents("Đội ngũ Lập Trình") == "Doi ngu Lap Trinh"
    expanded = expand_query("clb mm tuyen tv t10 ko")
    assert "multimedia" in expanded and "thanh vien" in expanded and "thang 10" in expanded
    assert "gcc" in expand_query("clb lam game cua hv")


def test_bm25_matches_query_without_diacritics(corpus):
    bm25 = BM25([d.text for d in corpus])
    assert int(bm25.scores("clb multimedia thanh lap ngay nao").argmax()) == 1
    assert bm25.scores("xyz").sum() == 0


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


def test_hybrid_without_rewrite_or_dense(corpus):
    llm = FakeLLM(lambda m: "")  # LLM trả rỗng: vẫn phải có câu trả lời hợp lệ
    pipeline = HybridPipeline(rewrite=False, dense=False, llm=llm)
    pipeline.index(corpus)
    out = pipeline.run("GCC ra doi nam nao")
    assert out.answer and out.retrieved_ids[0] == 3
    assert len(llm.calls) == 1

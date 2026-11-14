"""Backend của chatbot web (app/server.py): bọc các bước pipeline mà không đổi kết quả."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

from neorag.pipelines.base import RAGOutput

_spec = importlib.util.spec_from_file_location(
    "neorag_chat_server", Path(__file__).resolve().parents[1] / "app" / "server.py"
)
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)


@dataclass
class FakePlan:
    query: str
    rewritten: str = ""
    clubs: list[str] = field(default_factory=list)


class FakePipeline:
    """Giống cấu trúc HybridPipeline: run() gọi self.plan → self.retrieve → self.answer."""

    def plan(self, query: str) -> FakePlan:
        return FakePlan(query, rewritten=query.upper(), clubs=["gcc"])

    def retrieve(self, plan: FakePlan) -> list[int]:
        return [3, 1, 2]

    def answer(self, plan: FakePlan, candidates: list[int]) -> tuple[str, list[int]]:
        return f"trả lời: {plan.rewritten}", [1]

    def run(self, query: str) -> RAGOutput:
        plan = self.plan(query)
        ranked = self.retrieve(plan)
        text, used = self.answer(plan, ranked)
        return RAGOutput(answer=text, retrieved_ids=list(dict.fromkeys([*used, *ranked])))


def test_tracer_reports_steps_without_changing_output():
    plain = FakePipeline().run("clb game")
    traced_pipeline = FakePipeline()
    tracer = server.StepTracer(traced_pipeline)
    events = []
    tracer.on_step = lambda name, result, sec: events.append(server.step_event(name, result, sec))

    out = traced_pipeline.run("clb game")

    assert out == plain
    assert tracer.steps == ["plan", "retrieve", "answer"]
    assert [e["step"] for e in events] == ["plan", "retrieve", "answer"]
    assert events[0]["rewritten"] == "CLB GAME" and events[0]["clubs"] == ["gcc"]
    assert events[1]["n"] == 3
    assert events[2]["used"] == [1]


def test_tracer_skips_missing_steps():
    class RunOnly:
        def run(self, query: str) -> RAGOutput:
            return RAGOutput(answer="ok", retrieved_ids=[])

    assert server.StepTracer(RunOnly()).steps == []


def test_friendly_error_for_expired_key():
    class AuthenticationError(Exception):
        pass

    assert "hết hạn" in server.friendly_error(AuthenticationError("401"))
    assert server.friendly_error(ValueError("x")) == "ValueError: x"

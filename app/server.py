"""Chatbot riêng cho phần Demo Product: backend HTTP + giao diện web trong `app/web/`.

Chạy:  python app/server.py            → mở http://localhost:8000
       python app/server.py --port 8080

Chỉ dùng thư viện chuẩn của Python (không cần cài thêm). Pipeline mặc định là `HybridPipeline`,
đổi bằng biến môi trường NEORAG_PIPELINE="module.path:ClassName" như `app/chat.py`.

Server không sửa pipeline: nó chỉ bọc các bước `plan` / `retrieve` / `rerank` / `answer` (nếu
pipeline có) để báo tiến độ theo thời gian thực và lấy thông tin hiển thị (câu hỏi đã viết lại,
các đoạn LLM đã dùng). Câu trả lời và `retrieved_ids` vẫn do `pipeline.run()` tạo ra, giống hệt
lúc chạy `neorag run`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

WEB_DIR = Path(__file__).resolve().parent / "web"
PIPELINE_SPEC = os.getenv("NEORAG_PIPELINE", "neorag.pipelines.hybrid:HybridPipeline")
MAX_QUESTION_CHARS = 500
TRACED_STEPS = ("plan", "retrieve", "rerank", "answer")
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}
EXAMPLES = [  # câu tự soạn để demo, không lấy từ Public/Private Test
    {"q": "clb game co nhung su kien gi trong nam?", "tag": "Không dấu, viết tắt"},
    {"q": "ProPTIT và GCC khác nhau thế nào về cách tuyển thành viên?", "tag": "So sánh hai CLB"},
    {"q": "CLB Multimedia có phòng studio riêng không?", "tag": "Câu hỏi có/không"},
    {"q": "Năm nhất vào ProPTIT sẽ được học những gì?", "tag": "Cần nhiều đoạn"},
    {
        "q": "CLB Lập Trình PTIT có ký túc xá cho thành viên không?",
        "tag": "Không có trong tài liệu",
    },
]


class StepTracer:
    """Bọc các bước của pipeline để gọi `on_step(tên, kết_quả, giây)` mỗi khi một bước xong.

    Gán hàm bọc lên chính instance nên `run()` gọi `self.plan(...)` sẽ đi qua hàm bọc; kết quả
    trả về không đổi. Pipeline không có bước nào (ví dụ baseline) thì chỉ không có tiến độ.
    """

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.on_step: Callable[[str, Any, float], None] | None = None
        self.steps = [name for name in TRACED_STEPS if callable(getattr(pipeline, name, None))]
        for name in self.steps:
            setattr(pipeline, name, self._wrap(name, getattr(pipeline, name)))

    def _wrap(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def traced(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            if self.on_step is not None:
                self.on_step(name, result, time.perf_counter() - start)
            return result

        return traced


def step_event(name: str, result: Any, seconds: float) -> dict[str, Any]:
    """Thông tin hiển thị của một bước, chỉ lấy những trường có thật (getattr an toàn)."""
    event: dict[str, Any] = {"type": "step", "step": name, "ms": round(seconds * 1000)}
    if name == "plan":
        event["rewritten"] = getattr(result, "rewritten", "") or ""
        event["clubs"] = list(getattr(result, "clubs", []) or [])
        event["sub_queries"] = list(getattr(result, "sub_queries", []) or [])
    elif name in ("retrieve", "rerank") and isinstance(result, list):
        event["n"] = len(result)
    elif name == "answer" and isinstance(result, tuple) and len(result) == 2:
        event["used"] = [int(i) for i in result[1]]
    return event


def friendly_error(exc: Exception) -> str:
    """Lỗi gateway thường gặp → câu dễ hiểu; còn lại giữ nguyên tên lỗi để dễ tra."""
    name = type(exc).__name__
    if name == "AuthenticationError":
        return (
            "Key gateway sai hoặc đã hết hạn (kiểm tra OPENAI_API_KEY trong .env). "
            "Chỉ những câu đã có trong cache mới trả lời được."
        )
    if name == "RateLimitError":
        return "Gateway đang giới hạn tốc độ hoặc hết ngân sách, thử lại sau ít giây."
    if name in ("APIConnectionError", "APITimeoutError"):
        return "Không kết nối được gateway của BTC, kiểm tra mạng rồi thử lại."
    return f"{name}: {exc}"


def doc_dict(doc: Any) -> dict[str, Any]:
    return asdict(doc) if is_dataclass(doc) else dict(doc)


class ChatService:
    """Nạp pipeline ở luồng nền (server mở ngay, giao diện hiện 'đang nạp') và trả lời tuần tự."""

    def __init__(self, spec: str) -> None:
        self.spec = spec
        self.ready = False
        self.error = ""
        self.started = time.time()
        self.docs: dict[int, Any] = {}
        self.tracer: StepTracer | None = None
        self._lock = threading.Lock()  # một câu hỏi tại một thời điểm: tránh vượt rate limit

    def load(self) -> None:
        try:
            from neorag.data import load_corpus
            from neorag.pipelines import load_pipeline

            corpus = load_corpus()
            pipeline = load_pipeline(self.spec)
            pipeline.prepare()
            pipeline.index(corpus)
            self.docs = {d.id: d for d in corpus}
            self.tracer = StepTracer(pipeline)
            self.ready = True
            print(
                f"[server] Đã sẵn sàng sau {time.time() - self.started:.0f}s ({len(corpus)} đoạn)"
            )
        except Exception as exc:  # hiện lỗi lên giao diện thay vì làm sập server
            self.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    def health(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "error": self.error,
            "pipeline": self.spec,
            "docs": len(self.docs),
            "steps": self.tracer.steps if self.tracer else [],
        }

    def ask(self, question: str, emit: Callable[[dict[str, Any]], None]) -> None:
        assert self.tracer is not None
        with self._lock:
            used: list[int] = []

            def on_step(name: str, result: Any, seconds: float) -> None:
                event = step_event(name, result, seconds)
                used.extend(event.get("used", []))
                emit(event)

            self.tracer.on_step = on_step
            start = time.perf_counter()
            try:
                out = self.tracer.pipeline.run(question)
            finally:
                self.tracer.on_step = None
            cited = set(used)
            sources = [
                {**doc_dict(self.docs[i]), "cited": i in cited}
                for i in out.retrieved_ids
                if i in self.docs
            ]
            emit(
                {
                    "type": "done",
                    "answer": out.answer,
                    "sources": sources,
                    "latency_ms": round((time.perf_counter() - start) * 1000),
                }
            )


class Handler(BaseHTTPRequestHandler):
    service: ChatService  # gán trong main()
    server_version = "NeoRAGChat/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # gọn log: bỏ các request file tĩnh
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def _send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, rel: str) -> None:
        path = (WEB_DIR / rel).resolve()
        if WEB_DIR not in path.parents or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (tên do http.server quy định)
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json(self.service.health())
        elif path == "/api/examples":
            self._send_json(EXAMPLES)
        elif path.startswith("/api/doc/"):
            tail = path[len("/api/doc/") :]
            doc = self.service.docs.get(int(tail)) if tail.isdigit() else None
            if doc is None:
                self._send_json({"error": "Không có đoạn này"}, HTTPStatus.NOT_FOUND)
            else:
                self._send_json(doc_dict(doc))
        else:
            self._send_file("index.html" if path in ("", "/") else path.lstrip("/"))

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            question = str(json.loads(self.rfile.read(length) or b"{}").get("question", "")).strip()
        except (ValueError, AttributeError):
            self._send_json(
                {"error": 'Body phải là JSON {"question": ...}'}, HTTPStatus.BAD_REQUEST
            )
            return
        if not question:
            self._send_json({"error": "Câu hỏi đang trống"}, HTTPStatus.BAD_REQUEST)
            return
        if len(question) > MAX_QUESTION_CHARS:
            self._send_json(
                {"error": f"Câu hỏi dài quá {MAX_QUESTION_CHARS} ký tự"}, HTTPStatus.BAD_REQUEST
            )
            return
        if not self.service.ready:
            self._send_json(self.service.health(), HTTPStatus.SERVICE_UNAVAILABLE)
            return

        # Server-Sent Events: mỗi bước xong gửi một dòng `data: {...}`
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(event: dict[str, Any]) -> None:
            line = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()

        try:
            self.service.ask(question, emit)
        except (BrokenPipeError, ConnectionResetError):
            pass  # người dùng đóng tab giữa chừng
        except Exception as exc:
            traceback.print_exc()
            with contextlib.suppress(OSError):
                emit({"type": "error", "error": friendly_error(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Chatbot NeoRAG (giao diện web riêng)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):  # console Windows (cp1252) không in được tiếng Việt
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    service = ChatService(PIPELINE_SPEC)
    Handler.service = service
    threading.Thread(target=service.load, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[server] Pipeline: {PIPELINE_SPEC}")
    print(f"[server] Mở http://{args.host}:{args.port}  (Ctrl+C để dừng)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Dừng.")


if __name__ == "__main__":
    main()

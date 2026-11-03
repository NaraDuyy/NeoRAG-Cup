"""Embedding chạy local bằng sentence-transformers, cho ai muốn dùng model khác bge-m3 của gateway.

Cần cài thêm: python scripts/install.py --local
Lần chạy đầu sẽ tải model từ Hugging Face về máy (BAAI/bge-m3 khoảng 2.3 GB).
Model tự chạy trên GPU nếu PyTorch nhận được CUDA/MPS, ngược lại chạy trên CPU.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sqlite3
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

_load_lock = threading.Lock()


def is_downloaded(model_name: str) -> bool:
    """True nếu model đã có sẵn trên máy (thư mục local hoặc cache Hugging Face)."""
    if Path(model_name).exists():
        return True
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_name, local_files_only=True)
        return True
    except Exception:
        return False


def gpu_unused(device: str) -> bool:
    """True nếu máy có GPU NVIDIA nhưng model đang chạy CPU (thường do cài PyTorch bản CPU)."""
    return device == "cpu" and shutil.which("nvidia-smi") is not None


class _OnlyErrors(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR


def _silence_libraries() -> None:
    """Tắt log và thanh tiến độ phụ của torch/transformers để màn hình gọn.

    Phải gọi trước khi import sentence_transformers.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    # torch in "NOTE: Redirects are currently not supported in Windows or MacOs." khi import
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").addFilter(
        _OnlyErrors()
    )
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except ImportError:
        pass


@lru_cache(maxsize=4)
def _load(model_name: str):
    _silence_libraries()
    from sentence_transformers import SentenceTransformer

    # model đã có trên máy thì không kiểm tra mạng nữa: nạp nhanh hơn và chạy được khi offline
    return SentenceTransformer(model_name, local_files_only=is_downloaded(model_name))


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str, max_length: int):
    _silence_libraries()
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        model_name, max_length=max_length, local_files_only=is_downloaded(model_name)
    )


class LocalReranker:
    """Cross-encoder chạy local (ví dụ BAAI/bge-reranker-v2-m3) chấm từng cặp (câu hỏi, đoạn văn).

    Điểm được cache trong sqlite theo (model, câu hỏi, đoạn văn), nên chạy lại không tốn thời gian.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        cache_path: str | Path | None = ".cache/reranker.sqlite",
        batch_size: int = 8,
        max_length: int = 1024,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._lock = threading.Lock()
        self._db = None
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(cache_path, check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL)")

    def load(self):
        with _load_lock:
            return _load_cross_encoder(self.model_name, self.max_length)

    def _key(self, query: str, text: str) -> str:
        raw = "\x1f".join([self.model_name, str(self.max_length), query, text])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Điểm liên quan của từng đoạn với câu hỏi; càng cao càng liên quan."""
        keys = [self._key(query, t) for t in texts]
        scores: dict[str, float] = {}
        with self._lock:
            if self._db is not None:
                for key in keys:
                    row = self._db.execute("SELECT score FROM scores WHERE key = ?", (key,))
                    if hit := row.fetchone():
                        scores[key] = hit[0]
            missing = [(k, t) for k, t in zip(keys, texts, strict=True) if k not in scores]
            if missing:
                pairs = [(query, t) for _, t in missing]
                values = self.load().predict(
                    pairs, batch_size=self.batch_size, show_progress_bar=False
                )
                for (key, _), value in zip(missing, values, strict=True):
                    scores[key] = float(value)
                if self._db is not None:
                    self._db.executemany(
                        "INSERT OR REPLACE INTO scores VALUES (?, ?)",
                        [(k, scores[k]) for k, _ in missing],
                    )
                    self._db.commit()
        return [scores[k] for k in keys]


class LocalEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._lock = threading.Lock()  # bộ chấm gọi từ nhiều luồng

    def load(self):
        """Nạp model (tải về nếu chưa có). Gọi nhiều lần cũng chỉ nạp một lần."""
        with _load_lock:
            return _load(self.model_name)

    @property
    def device(self) -> str:
        return str(self.load().device)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Trả về ma trận (len(texts), dim) đã chuẩn hóa L2."""
        model = self.load()
        with self._lock:
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype=np.float32)

from __future__ import annotations

import os
import re
from hashlib import sha1
from pathlib import Path
from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    name: str

    @property
    def dimensions(self) -> int: ...

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbeddings:
    """Lazy local dense embeddings backed by a real transformer model."""

    def __init__(self, model_name: str, display_name: str | None = None) -> None:
        self.model_name = model_name
        self.name = f"sentence_transformers:{display_name or model_name}"
        self._model = None
        self._dimensions = 0

    def _load(self):
        if self._model is None:
            # Hugging Face Xet can stall behind local SOCKS proxies. Regular
            # HTTPS downloads are slower but work consistently in local and CI
            # environments; callers can override this env before startup.
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            dimension_getter = (
                self._model.get_embedding_dimension
                if hasattr(self._model, "get_embedding_dimension")
                else self._model.get_sentence_embedding_dimension
            )
            self._dimensions = int(dimension_getter())
        return self._model

    @property
    def dimensions(self) -> int:
        self._load()
        return self._dimensions

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        instruction = os.getenv(
            "AI_AGENT_EMBEDDING_QUERY_INSTRUCTION",
            "为这个句子生成表示以用于检索相关文章：",
        )
        prepared = [f"{instruction}{text}" if instruction else text for text in texts]
        return self._encode(prepared)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=max(1, int(os.getenv("AI_AGENT_EMBEDDING_BATCH_SIZE", "32"))),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class DeterministicTestEmbeddings:
    """Small deterministic encoder stub for unit tests only.

    It is deliberately gated behind ``AI_AGENT_ALLOW_TEST_EMBEDDINGS`` so the
    application cannot silently present synthetic vectors as real embeddings.
    """

    name = "test_stub"

    def __init__(self, dimensions: int = 384) -> None:
        if os.getenv("AI_AGENT_ALLOW_TEST_EMBEDDINGS") != "1":
            raise RuntimeError("test embeddings require AI_AGENT_ALLOW_TEST_EMBEDDINGS=1")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            lowered = text.lower()
            tokens = re.findall(r"[a-z][a-z0-9_-]+|[\u4e00-\u9fff]", lowered)
            tokens.extend(lowered[index : index + 3] for index in range(max(0, len(lowered) - 2)))
            for token in tokens:
                digest = sha1(token.encode("utf-8")).digest()
                column = int.from_bytes(digest[:4], "big") % self.dimensions
                vectors[row, column] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def build_embedding_provider(preference: str | None = None) -> EmbeddingProvider:
    if preference:
        selected = preference
    else:
        selected = os.getenv("AI_AGENT_EMBEDDING_PROVIDER", "sentence_transformers")
    selected = selected.strip().lower()
    if selected in {"test", "test_stub"}:
        return DeterministicTestEmbeddings()
    if selected not in {"sentence_transformers", "sentence-transformers", "local"}:
        raise ValueError(f"unsupported embedding provider: {selected}")
    configured_model = os.getenv("AI_AGENT_EMBEDDING_MODEL")
    display_name = configured_model or "BAAI/bge-small-zh-v1.5"
    local_model = Path(__file__).resolve().parents[1] / ".models" / "bge-small-zh-v1.5"
    model_name = (
        str(local_model)
        if configured_model is None and (local_model / "model.safetensors").exists()
        else display_name
    )
    return SentenceTransformerEmbeddings(model_name, display_name=display_name)

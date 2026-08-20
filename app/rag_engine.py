from __future__ import annotations

import os
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np

from .context_engine import ContextBundle, ContextPacker, EvidenceGraph
from .embeddings import EmbeddingProvider, build_embedding_provider
from .rerankers import build_reranker

SUPPORTED_EXTENSIONS = {".md", ".txt"}


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str
    source: str
    text: str


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    vector_score: float
    keyword_score: float
    title_score: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self.chunk)
        payload.update(
            {
                "score": round(float(self.score), 4),
                "vector_score": round(float(self.vector_score), 4),
                "keyword_score": round(float(self.keyword_score), 4),
                "title_score": round(float(self.title_score), 4),
                "snippet": compact_text(self.chunk.text, 220),
            }
        )
        return payload


@dataclass
class VectorCandidate:
    chunk_index: int
    vector_score: float


class DenseVectorStore:
    """In-memory cosine search over real dense document embeddings.

    ``build`` reuses embeddings cached by ``chunk.id`` when ``reuse_ids`` is
    given, so reloads only encode the delta (new or changed documents) instead
    of re-encoding the whole corpus.
    """

    def __init__(self, embeddings: EmbeddingProvider) -> None:
        self.embeddings = embeddings
        self.name = f"dense_memory:{embeddings.name}"
        self.matrix = np.empty((0, 0), dtype=np.float32)
        self._rows_by_id: dict[str, np.ndarray] = {}

    def build(
        self,
        chunks: list[Chunk],
        index_views: list["_IndexView"],
        reuse_ids: set[str] | None = None,
    ) -> None:
        rows: list[np.ndarray | None] = [None] * len(chunks)
        to_encode: list[tuple[int, str]] = []
        for idx, (chunk, view) in enumerate(zip(chunks, index_views)):
            if reuse_ids and chunk.id in reuse_ids:
                cached = self._rows_by_id.get(chunk.id)
                if cached is not None:
                    rows[idx] = cached
                    continue
            to_encode.append((idx, view.text_for_index))
        if to_encode:
            encoded = self.embeddings.encode_documents([text for _, text in to_encode])
            for (idx, _), row in zip(to_encode, encoded):
                rows[idx] = row
        if not rows:
            self.matrix = np.empty((0, 0), dtype=np.float32)
            self._rows_by_id = {}
            return
        filled = [row for row in rows if row is not None]
        if len(filled) != len(rows):
            raise RuntimeError("dense index rows do not match chunks after build")
        self.matrix = np.vstack(filled).astype(np.float32)
        self._rows_by_id = {chunk.id: self.matrix[i] for i, chunk in enumerate(chunks)}

    def search(self, query: str, top_n: int) -> list[VectorCandidate]:
        if self.matrix.size == 0:
            return []
        query_vector = self.embeddings.encode_queries([query])[0]
        scores = self.matrix @ query_vector
        indices = np.argsort(scores)[::-1][:top_n]
        return [
            VectorCandidate(chunk_index=int(index), vector_score=max(0.0, float(scores[index])))
            for index in indices
        ]


class ChromaVectorStore:
    """Persistent Chroma index populated with the same real dense embeddings.

    ``build`` upserts only new/changed chunks and deletes stale ids instead of
    dropping and recreating the whole collection, so reloads are incremental.
    The collection is tagged with the embedding model so a model change forces
    a full rebuild instead of mixing incompatible vectors.
    """

    def __init__(self, persist_dir: Path, embeddings: EmbeddingProvider):
        self.persist_dir = persist_dir
        self.embeddings = embeddings
        self.name = f"chroma:{embeddings.name}"
        self.collection: Any = None  # lazy chromadb.Collection, loaded in build()
        self.chunk_count = 0

    def build(
        self,
        chunks: list[Chunk],
        index_views: list["_IndexView"],
        reuse_ids: set[str] | None = None,
    ) -> None:
        import chromadb

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        collection_name = "engineering_evidence"
        model_tag = f"{self.embeddings.name}:{self.embeddings.dimensions}"
        try:
            collection = client.get_collection(name=collection_name)
            if (collection.metadata or {}).get("model") != model_tag:
                client.delete_collection(collection_name)
                collection = client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine", "model": model_tag},
                )
        except Exception:
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine", "model": model_tag},
            )
        self.collection = collection
        self.chunk_count = len(chunks)

        view_by_id = {chunk.id: view.text_for_index for chunk, view in zip(chunks, index_views)}
        index_by_id = {chunk.id: idx for idx, chunk in enumerate(chunks)}
        source_by_id = {chunk.id: chunk.source for chunk in chunks}

        existing_ids = set(collection.get(include=[])["ids"])
        target_ids = {chunk.id for chunk in chunks}

        stale = existing_ids - target_ids
        if stale:
            collection.delete(ids=list(stale))

        # Unchanged chunks already persisted: refresh only the index metadata
        # because positions shift when other documents are added or removed.
        keep_ids = [
            chunk.id
            for chunk in chunks
            if chunk.id in existing_ids and reuse_ids and chunk.id in reuse_ids
        ]
        if keep_ids:
            collection.update(
                ids=keep_ids,
                metadatas=[
                    {"chunk_index": index_by_id[cid], "source": source_by_id[cid]}
                    for cid in keep_ids
                ],
            )

        # New chunks and chunks of changed documents are encoded as a delta.
        encode_chunks = [
            chunk
            for chunk in chunks
            if chunk.id not in existing_ids or not (reuse_ids and chunk.id in reuse_ids)
        ]
        if encode_chunks:
            corpus = [view_by_id[chunk.id] for chunk in encode_chunks]
            embeddings = self.embeddings.encode_documents(corpus).tolist()
            collection.upsert(
                ids=[chunk.id for chunk in encode_chunks],
                documents=corpus,
                embeddings=embeddings,
                metadatas=[
                    {"chunk_index": index_by_id[chunk.id], "source": chunk.source}
                    for chunk in encode_chunks
                ],
            )

    def search(self, query: str, top_n: int) -> list[VectorCandidate]:
        if self.collection is None or self.chunk_count == 0:
            return []

        embedding = self.embeddings.encode_queries([query])[0].tolist()
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_n, self.chunk_count),
            include=["distances", "metadatas"],
        )
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        candidates: list[VectorCandidate] = []
        for distance, metadata in zip(distances, metadatas):
            chunk_index = int(metadata["chunk_index"])
            vector_score = max(0.0, 1.0 - float(distance))
            candidates.append(VectorCandidate(chunk_index=chunk_index, vector_score=vector_score))
        return candidates


class KnowledgeBase:
    """Engineering-evidence RAG engine with swappable vector store backends."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.chunks: list[Chunk] = []
        self.embedding_provider = build_embedding_provider()
        self.vector_store = self._select_vector_store()
        self.reranker = build_reranker()
        self.vector_backend_error: str | None = None
        self.last_loaded_at = 0.0
        self._lock = threading.Lock()
        self._stats_cache: dict[str, Any] | None = None
        self._file_snapshots: dict[str, str] = {}
        self.evidence_graph = EvidenceGraph([])
        self._ready = False
        self.load()

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready

    def _select_vector_store(self):
        preference = os.getenv("AI_AGENT_VECTOR_BACKEND", "memory").strip().lower()
        if preference == "chroma":
            return ChromaVectorStore(
                self.data_dir.parent / "vector_store" / "chroma",
                self.embedding_provider,
            )
        if preference not in {"memory", "dense", "auto"}:
            raise ValueError(
                "AI_AGENT_VECTOR_BACKEND must be memory or chroma"
            )
        return DenseVectorStore(self.embedding_provider)

    def load(self) -> None:
        with self._lock:
            self._reload_locked()

    def _reload_locked(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        for file_path in sorted(self.data_dir.rglob("*")):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if any(part.startswith(".") for part in file_path.relative_to(self.data_dir).parts):
                continue
            files[str(file_path)] = _file_digest(file_path)

        changed_paths = {
            path for path, digest in files.items() if self._file_snapshots.get(path) != digest
        }
        removed_paths = set(self._file_snapshots) - set(files)
        if not changed_paths and not removed_paths and self._ready:
            return  # nothing changed; keep the existing index untouched

        chunks: list[Chunk] = []
        for path in sorted(files):
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            file_chunks = chunk_document(Path(path), text)
            chunks.extend(file_chunks)

        # Chunk ids include normalized content, so unchanged chunks from a
        # modified document can safely keep their existing embedding row.
        previous_ids = {chunk.id for chunk in self.chunks}
        reuse_ids = {chunk.id for chunk in chunks if chunk.id in previous_ids}
        index_views = _chunk_index_views(chunks)
        try:
            self.vector_backend_error = None
            self.vector_store.build(chunks, index_views, reuse_ids=reuse_ids)
        except Exception as exc:
            self.vector_backend_error = f"{type(exc).__name__}: {exc}"
            if isinstance(self.vector_store, DenseVectorStore):
                raise
            self.vector_store = DenseVectorStore(self.embedding_provider)
            self.vector_store.build(chunks, index_views)
        self.chunks = chunks
        self.evidence_graph = EvidenceGraph(chunks)
        self._file_snapshots = files
        self.last_loaded_at = time.time()
        self._stats_cache = None
        self._ready = True

    def stats(self) -> dict[str, Any]:
        with self._lock:
            if self._stats_cache is None:
                doc_ids = {c.doc_id for c in self.chunks}
                sections = {c.section for c in self.chunks}
                self._stats_cache = {
                    "documents": len(doc_ids),
                    "chunks": len(self.chunks),
                    "sections": len(sections),
                    "data_dir": str(self.data_dir),
                    "last_loaded_at": self.last_loaded_at,
                    "embedding_model": self.embedding_provider.name,
                    "embedding_dimensions": self.embedding_provider.dimensions,
                    "evidence_relations": len(self.evidence_graph.relations),
                }
            # Runtime state is always read live so a mid-flight reranker
            # fallback is still reflected even when the doc stats are cached.
            return {
                **self._stats_cache,
                "vector_backend": self.vector_store.name,
                "vector_backend_error": self.vector_backend_error,
                "reranker_requested": self.reranker.requested_name,
                "reranker": self.reranker.name,
                "reranker_error": self.reranker.last_error,
                "reranker_degraded": self.reranker.degraded,
                "reranker_fallback_count": self.reranker.fallback_count,
            }

    def documents(self) -> list[dict[str, Any]]:
        with self._lock:
            chunks = self.chunks
        grouped: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            item = grouped.setdefault(
                chunk.doc_id,
                {
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "chunks": 0,
                    "sections": set(),
                    "deletable": True,
                },
            )
            item["chunks"] += 1
            item["sections"].add(chunk.section)
        docs = []
        for item in grouped.values():
            item["sections"] = sorted(item["sections"])
            docs.append(item)
        return sorted(docs, key=lambda x: x["title"])

    def document_path(self, doc_id: str) -> Path | None:
        for file_path in sorted(self.data_dir.rglob("*")):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if any(part.startswith(".") for part in file_path.relative_to(self.data_dir).parts):
                continue
            if stable_doc_id(file_path) == doc_id:
                return file_path
        return None

    def soft_delete_document(self, doc_id: str) -> Path | None:
        """Move a document to the recoverable tombstone area and rebuild the delta."""

        path = self.document_path(doc_id)
        if path is None:
            return None
        trash_dir = self.data_dir / ".trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        target = trash_dir / path.name
        if target.exists():
            target = trash_dir / f"{path.stem}-{int(time.time())}{path.suffix}"
        shutil.move(str(path), str(target))
        self.load()
        return target

    def deleted_documents(self) -> list[dict[str, str]]:
        trash_dir = self.data_dir / ".trash"
        if not trash_dir.exists():
            return []
        return [
            {"doc_id": stable_doc_id(path), "source": path.name, "path": str(path)}
            for path in sorted(trash_dir.iterdir())
            if path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    def restore_document(self, source: str) -> Path | None:
        trash_dir = self.data_dir / ".trash"
        path = trash_dir / Path(source).name
        if not path.exists() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return None
        target = self.data_dir / path.name
        if target.exists():
            return None
        shutil.move(str(path), str(target))
        self.load()
        return target

    def search(self, query: str, top_k: int = 6) -> list[SearchHit]:
        # Snapshot the mutable index under the lock so a concurrent reload
        # cannot swap chunks/vector_store mid-search (index-out-of-range).
        with self._lock:
            chunks = self.chunks
            vector_store = self.vector_store
        if not chunks:
            return []

        expanded_query = expand_query(query)
        vector_candidates = vector_store.search(expanded_query, top_n=max(top_k * 4, 24))
        query_terms = extract_terms(expanded_query)

        # Single pass over the corpus: compute keyword/title scores once and
        # keep every chunk that is a vector candidate or has a direct term hit.
        scored: dict[int, tuple[float, float]] = {}
        for idx, chunk in enumerate(chunks):
            keyword_score = keyword_overlap(query_terms, chunk.text)
            title_score = title_overlap(query_terms, f"{chunk.title} {chunk.section}")
            if keyword_score > 0 or title_score > 0:
                scored[idx] = (keyword_score, title_score)

        vector_scores = {c.chunk_index: c.vector_score for c in vector_candidates}
        if not scored and not vector_scores:
            return []

        hits: list[SearchHit] = []
        for idx, vector_score in vector_scores.items():
            keyword_score, title_score = scored.get(idx, (0.0, 0.0))
            score = 0.72 * vector_score + 0.20 * keyword_score + 0.08 * title_score
            hits.append(
                SearchHit(
                    chunk=chunks[idx],
                    score=score,
                    vector_score=vector_score,
                    keyword_score=keyword_score,
                    title_score=title_score,
                )
            )
        # Chunks matched only by exact terms still join the pool with a zero
        # vector score, mirroring the previous candidate-set behaviour.
        for idx, (keyword_score, title_score) in scored.items():
            if idx not in vector_scores:
                hits.append(
                    SearchHit(
                        chunk=chunks[idx],
                        score=0.20 * keyword_score + 0.08 * title_score,
                        vector_score=0.0,
                        keyword_score=keyword_score,
                        title_score=title_score,
                    )
                )

        hits.sort(key=lambda h: h.score, reverse=True)
        return self.reranker.rerank(query, hits[: max(top_k * 3, 10)])[:top_k]

    def build_context(
        self,
        query: str,
        *,
        top_k: int = 6,
        token_budget: int = 1800,
        graph_expansion: int = 2,
    ) -> ContextBundle:
        """Run hybrid retrieval, bounded graph expansion and context packing."""

        initial = self.search(query, top_k=max(top_k, graph_expansion + top_k))
        expanded, relations = self.evidence_graph.expand(initial, limit=graph_expansion)
        packer = ContextPacker(token_budget=token_budget)
        return packer.pack(expanded, relations)


@dataclass
class _IndexView:
    text_for_index: str


def _chunk_index_views(chunks: list[Chunk]) -> list[_IndexView]:
    return [
        _IndexView(text_for_index=f"{chunk.title}\n{chunk.section}\n{chunk.text}")
        for chunk in chunks
    ]


def chunk_document(file_path: Path, text: str) -> list[Chunk]:
    normalized = normalize_text(text)
    title = extract_title(normalized, file_path)
    doc_id = stable_doc_id(file_path)
    sections = split_sections(normalized)
    chunks: list[Chunk] = []

    occurrences: dict[str, int] = {}
    for section_title, section_text in sections:
        for piece in split_to_chunks(section_text):
            if len(piece.strip()) < 20:
                continue
            normalized_piece = piece.strip()
            digest = sha1(
                f"{section_title}\n{normalized_piece}".encode("utf-8")
            ).hexdigest()[:12]
            occurrences[digest] = occurrences.get(digest, 0) + 1
            chunks.append(
                Chunk(
                    id=f"{doc_id}-{digest}-{occurrences[digest]}",
                    doc_id=doc_id,
                    title=title,
                    section=section_title or title,
                    source=file_path.name,
                    text=normalized_piece,
                )
            )
    return chunks


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_title(text: str, file_path: Path) -> str:
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("#"):
            return cleaned.lstrip("#").strip()
    return file_path.stem.replace("_", " ").replace("-", " ").strip().title()


def stable_doc_id(file_path: Path) -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "-", file_path.stem, flags=re.UNICODE)
    slug = slug.strip("-_").lower()
    if slug:
        return slug
    return f"doc-{sha1(file_path.name.encode('utf-8')).hexdigest()[:10]}"


def _file_digest(file_path: Path) -> str:
    """Content hash used to detect changed documents for incremental reloads."""
    return sha1(file_path.read_bytes()).hexdigest()


def split_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("#"):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line.strip().lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))
    if not sections:
        return [("", text)]
    return [(title, "\n".join(part).strip()) for title, part in sections if "\n".join(part).strip()]


def split_to_chunks(text: str, max_chars: int = 720, overlap: int = 90) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
            continue
        start = 0
        while start < len(paragraph):
            end = min(len(paragraph), start + max_chars)
            chunks.append(paragraph[start:end])
            if end == len(paragraph):
                break
            start = max(0, end - overlap)
        current = ""
    if current:
        chunks.append(current)
    return chunks


def expand_query(query: str) -> str:
    expansions = {
        "rag": "retrieval augmented generation 检索增强生成 知识库 召回 引用",
        "agent": "tool use planning memory 工具调用 任务规划 多轮记忆 智能体",
        "embedding": "向量 表征 语义相似度 vector embeddings",
        "rerank": "重排序 相关性排序 cross encoder rank",
        "openai": "GPT ChatGPT 大模型 LLM Embedding API 可插拔模型供应商",
        "llm": "大语言模型 大模型 生成 回答 OpenAI DeepSeek Qwen 混元",
        "大模型": "LLM 生成 回答 OpenAI DeepSeek Qwen 混元",
        "prompt": "提示词 Prompt 工程 输出结构 约束 示例",
        "知识库": "文档 资料 chunk 分段 metadata source citation",
        "变更": "影响面 调用方 兼容性 状态 数据 依赖 回滚",
        "故障": "日志 原始错误 失败阶段 根因假设 重试 回退 恢复",
        "审查": "PR review 风险 测试缺口 错误处理 兼容性",
        "验证": "baseline oracle 测试 故障注入 停止条件 回滚",
        "部署": "FastAPI 前端 服务 API Docker screen uvicorn",
    }
    text = query
    lower = query.lower()
    for key, value in expansions.items():
        if key.isascii() and key.isalpha():
            # Word-boundary match for ASCII keys: "rag" must not fire on
            # substrings like "storage" or "coverage".
            if re.search(rf"(?<![a-z0-9]){re.escape(key.lower())}(?![a-z0-9])", lower):
                text += " " + value
        elif key.lower() in lower or key in query:
            text += " " + value
    return text


def extract_terms(text: str) -> set[str]:
    lower = text.lower()
    english = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{1,}", lower)
    chinese_sequences = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    chinese: list[str] = []
    for sequence in chinese_sequences:
        # Chinese queries do not contain whitespace, so keeping only the full
        # sequence turns an entire question into one impossible keyword. Short
        # n-grams retain domain phrases such as “自然语言处理” and “命名实体”.
        upper = min(6, len(sequence))
        for width in range(2, upper + 1):
            chinese.extend(
                sequence[index : index + width]
                for index in range(len(sequence) - width + 1)
            )
    short = re.findall(r"[A-Z]{2,}", text)
    terms = set(english + chinese + [s.lower() for s in short])
    stop = {
        "什么",
        "怎么",
        "如何",
        "一下",
        "这个",
        "那个",
        "项目",
        "系统",
        "可以",
        "需要",
        "资料",
        "根据",
        "总结",
        "重点",
    }
    return {t for t in terms if t not in stop}


def keyword_overlap(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    lower = text.lower()
    matched = sum(1 for term in query_terms if term in lower or term in text)
    return matched / max(len(query_terms), 1)


def title_overlap(query_terms: set[str], title: str) -> float:
    if not query_terms:
        return 0.0
    lower = title.lower()
    matched = sum(1 for term in query_terms if term in lower or term in title)
    return min(1.0, matched / 3)


def compact_text(text: str, limit: int = 220) -> str:
    one_line = re.sub(r"\s+", " ", text).strip()
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 1].rstrip() + "…"

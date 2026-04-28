from __future__ import annotations

import re
import time
import os
from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


class TfidfVectorStore:
    """Zero-config vector backend for local demos and CI."""

    name = "tfidf_local"

    def __init__(self) -> None:
        self._char_vectorizer: TfidfVectorizer | None = None
        self._word_vectorizer: TfidfVectorizer | None = None
        self._char_matrix = None
        self._word_matrix = None

    def build(self, chunks: list[Chunk], index_views: list["_IndexView"]) -> None:
        corpus = [view.text_for_index for view in index_views]
        if not corpus:
            self._char_vectorizer = None
            self._word_vectorizer = None
            self._char_matrix = None
            self._word_matrix = None
            return

        self._char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=1,
            sublinear_tf=True,
            norm="l2",
        )
        self._word_vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b[\w\-]+\b",
            ngram_range=(1, 2),
            lowercase=True,
            min_df=1,
            sublinear_tf=True,
            norm="l2",
        )
        self._char_matrix = self._char_vectorizer.fit_transform(corpus)
        self._word_matrix = self._word_vectorizer.fit_transform(corpus)

    def search(self, query: str, top_n: int) -> list[VectorCandidate]:
        if not self._char_vectorizer or self._char_matrix is None:
            return []

        q_char = self._char_vectorizer.transform([query])
        char_scores = cosine_similarity(q_char, self._char_matrix).ravel()

        if self._word_vectorizer and self._word_matrix is not None:
            q_word = self._word_vectorizer.transform([query])
            word_scores = cosine_similarity(q_word, self._word_matrix).ravel()
        else:
            word_scores = np.zeros(len(char_scores))

        candidates = [
            VectorCandidate(
                chunk_index=idx,
                vector_score=0.68 * float(char_scores[idx]) + 0.32 * float(word_scores[idx]),
            )
            for idx in range(len(char_scores))
        ]
        candidates.sort(key=lambda item: item.vector_score, reverse=True)
        return candidates[:top_n]


class ChromaVectorStore:
    """Persistent vector database backend.

    Chroma is used when ``AI_AGENT_VECTOR_BACKEND=chroma`` or when the default
    ``auto`` mode finds chromadb installed. A fixed-size HashingVectorizer keeps
    the backend self-contained for a student demo; swapping it for bge/OpenAI
    embeddings only requires changing the embedding function here.
    """

    name = "chroma_hashing"

    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.collection = None
        self.chunk_count = 0
        self.vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            n_features=768,
            alternate_sign=False,
            norm="l2",
        )

    def build(self, chunks: list[Chunk], index_views: list["_IndexView"]) -> None:
        import chromadb

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        collection_name = "course_chunks"
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        self.collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.chunk_count = len(chunks)
        if not chunks:
            return

        corpus = [view.text_for_index for view in index_views]
        embeddings = self.vectorizer.transform(corpus).toarray().astype("float32").tolist()
        ids = [f"{chunk.id}-{idx}" for idx, chunk in enumerate(chunks)]
        metadatas = [{"chunk_index": idx, "source": chunk.source} for idx, chunk in enumerate(chunks)]
        self.collection.add(
            ids=ids,
            documents=corpus,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(self, query: str, top_n: int) -> list[VectorCandidate]:
        if self.collection is None or self.chunk_count == 0:
            return []

        embedding = self.vectorizer.transform([query]).toarray().astype("float32")[0].tolist()
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
    """Course-material RAG engine with swappable vector store backends."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.chunks: list[Chunk] = []
        self.vector_store = self._select_vector_store()
        self.vector_backend_error: str | None = None
        self.last_loaded_at = 0.0
        self.load()

    def _select_vector_store(self):
        preference = os.getenv("AI_AGENT_VECTOR_BACKEND", "auto").strip().lower()
        if preference in {"auto", "chroma"}:
            return ChromaVectorStore(self.data_dir.parent / "vector_store" / "chroma")
        return TfidfVectorStore()

    def load(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = []
        for file_path in sorted(self.data_dir.rglob("*")):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            self.chunks.extend(chunk_document(file_path, text))

        index_views = _chunk_index_views(self.chunks)
        try:
            self.vector_backend_error = None
            self.vector_store.build(self.chunks, index_views)
        except Exception as exc:
            self.vector_backend_error = f"{type(exc).__name__}: {exc}"
            self.vector_store = TfidfVectorStore()
            self.vector_store.build(self.chunks, index_views)
        self.last_loaded_at = time.time()

    def stats(self) -> dict[str, Any]:
        doc_ids = {c.doc_id for c in self.chunks}
        sections = {c.section for c in self.chunks}
        return {
            "documents": len(doc_ids),
            "chunks": len(self.chunks),
            "sections": len(sections),
            "data_dir": str(self.data_dir),
            "last_loaded_at": self.last_loaded_at,
            "vector_backend": self.vector_store.name,
            "vector_backend_error": self.vector_backend_error,
        }

    def documents(self) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for chunk in self.chunks:
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
            if stable_doc_id(file_path) == doc_id:
                return file_path
        return None

    def search(self, query: str, top_k: int = 6) -> list[SearchHit]:
        if not self.chunks:
            return []

        expanded_query = expand_query(query)
        vector_candidates = self.vector_store.search(expanded_query, top_n=max(top_k * 4, 24))
        candidate_scores = {
            candidate.chunk_index: candidate.vector_score for candidate in vector_candidates
        }
        query_terms = extract_terms(expanded_query)

        # Keep exact terms in the candidate pool so IDs, names, and short
        # professional nouns are not lost by pure vector similarity.
        for idx, chunk in enumerate(self.chunks):
            keyword_score = keyword_overlap(query_terms, chunk.text)
            title_score = title_overlap(query_terms, f"{chunk.title} {chunk.section}")
            if keyword_score > 0 or title_score > 0:
                candidate_scores.setdefault(idx, 0.0)

        if not candidate_scores:
            return []

        hits: list[SearchHit] = []
        for idx, vector_score in candidate_scores.items():
            chunk = self.chunks[idx]
            keyword_score = keyword_overlap(query_terms, chunk.text)
            title_score = title_overlap(query_terms, f"{chunk.title} {chunk.section}")
            score = 0.72 * vector_score + 0.20 * keyword_score + 0.08 * title_score
            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=score,
                    vector_score=vector_score,
                    keyword_score=keyword_score,
                    title_score=title_score,
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        return rerank_hits(query, hits[: max(top_k * 3, 10)])[:top_k]


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

    chunk_no = 0
    for section_title, section_text in sections:
        for piece in split_to_chunks(section_text):
            if len(piece.strip()) < 20:
                continue
            chunk_no += 1
            chunks.append(
                Chunk(
                    id=f"{doc_id}-{chunk_no:03d}",
                    doc_id=doc_id,
                    title=title,
                    section=section_title or title,
                    source=file_path.name,
                    text=piece.strip(),
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
        "出题": "选择题 判断题 练习题 自测 题干 选项 答案 解析",
        "选择题": "出题 练习题 自测 题干 选项 答案 解析",
        "简历": "岗位 匹配 项目亮点 负责内容 技术栈 成果",
        "腾讯": "AI原生工程师 产品业务 系统开发 Agent RAG",
        "部署": "FastAPI 前端 服务 API Docker screen uvicorn",
    }
    text = query
    lower = query.lower()
    for key, value in expansions.items():
        if key.lower() in lower or key in query:
            text += " " + value
    return text


def extract_terms(text: str) -> set[str]:
    lower = text.lower()
    english = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{1,}", lower)
    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    short = re.findall(r"[A-Z]{2,}", text)
    terms = set(english + chinese + [s.lower() for s in short])
    stop = {"什么", "怎么", "如何", "一下", "这个", "那个", "项目", "系统", "可以", "需要"}
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


def rerank_hits(query: str, hits: list[SearchHit]) -> list[SearchHit]:
    query_terms = extract_terms(expand_query(query))
    for hit in hits:
        dense_bonus = min(0.08, len(query_terms & extract_terms(hit.chunk.text)) * 0.015)
        structure_bonus = 0.04 if any(k in hit.chunk.section.lower() for k in ["agent", "rag", "评估", "架构"]) else 0.0
        hit.score = float(hit.score + dense_bonus + structure_bonus)
    return sorted(hits, key=lambda h: h.score, reverse=True)


def compact_text(text: str, limit: int = 220) -> str:
    one_line = re.sub(r"\s+", " ", text).strip()
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 1].rstrip() + "…"

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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


class KnowledgeBase:
    """Small local RAG engine with hybrid retrieval and reranking.

    It intentionally avoids heavyweight vector database dependencies so the
    project can run in a classroom or interview demo environment immediately.
    The interface is kept close to a real vector store so FAISS, Chroma, Milvus,
    or Elasticsearch can replace this class later.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.chunks: list[Chunk] = []
        self._char_vectorizer: TfidfVectorizer | None = None
        self._word_vectorizer: TfidfVectorizer | None = None
        self._char_matrix = None
        self._word_matrix = None
        self.last_loaded_at = 0.0
        self.load()

    def load(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = []
        for file_path in sorted(self.data_dir.rglob("*")):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            self.chunks.extend(chunk_document(file_path, text))

        corpus = [c.text_for_index for c in _chunk_index_views(self.chunks)]
        if corpus:
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
        else:
            self._char_vectorizer = None
            self._word_vectorizer = None
            self._char_matrix = None
            self._word_matrix = None
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
        if not self.chunks or not self._char_vectorizer or self._char_matrix is None:
            return []

        expanded_query = expand_query(query)
        q_char = self._char_vectorizer.transform([expanded_query])
        char_scores = cosine_similarity(q_char, self._char_matrix).ravel()

        if self._word_vectorizer and self._word_matrix is not None:
            q_word = self._word_vectorizer.transform([expanded_query])
            word_scores = cosine_similarity(q_word, self._word_matrix).ravel()
        else:
            word_scores = np.zeros(len(self.chunks))

        query_terms = extract_terms(expanded_query)
        hits: list[SearchHit] = []
        for idx, chunk in enumerate(self.chunks):
            vector_score = 0.68 * float(char_scores[idx]) + 0.32 * float(word_scores[idx])
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
    return re.sub(r"[^a-zA-Z0-9]+", "-", file_path.stem).strip("-").lower()


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

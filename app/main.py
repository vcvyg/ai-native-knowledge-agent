from __future__ import annotations

import base64
import io
import time
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import AgentResponse, KnowledgeAgent, ToolCall
from .rag_engine import KnowledgeBase


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "knowledge_base"
WEB_DIR = ROOT / "web"


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1200)
    session_id: str | None = None
    top_k: int = Field(default=6, ge=1, le=12)


class DocumentRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)
    content: str = Field(..., min_length=20)
    source: str | None = Field(default=None, max_length=120)


class DocumentUploadRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=180)
    content_base64: str = Field(..., min_length=8)
    title: str | None = Field(default=None, max_length=120)


app = FastAPI(
    title="AI Course Study Agent",
    description="Course-material RAG + Learning Agent demo for AI native engineering roles.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

kb = KnowledgeBase(DATA_DIR)
agent = KnowledgeAgent(kb)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "ai-course-study-agent",
        "time": time.time(),
        "kb": kb.stats(),
    }


@app.get("/api/kb/stats")
def kb_stats() -> dict[str, Any]:
    return kb.stats()


@app.get("/api/kb/documents")
def kb_documents() -> list[dict[str, Any]]:
    return kb.documents()


@app.post("/api/kb/reload")
def reload_kb() -> dict[str, Any]:
    kb.load()
    return {"success": True, "kb": kb.stats()}


@app.post("/api/kb/documents")
def add_document(payload: DocumentRequest) -> dict[str, Any]:
    filename = unique_filename(safe_filename(payload.source or payload.title), ".md")
    target = DATA_DIR / filename
    header = f"# {payload.title.strip()}\n\n"
    target.write_text(header + payload.content.strip() + "\n", encoding="utf-8")
    kb.load()
    return {"success": True, "document": filename, "kb": kb.stats()}


@app.post("/api/kb/upload")
def upload_document(payload: DocumentUploadRequest) -> dict[str, Any]:
    try:
        raw = decode_base64_payload(payload.content_base64)
        text = extract_uploaded_text(payload.filename, raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if len(text.strip()) < 20:
        raise HTTPException(status_code=400, detail="文档内容过短或解析失败")

    title = payload.title or Path(payload.filename).stem
    filename = unique_filename(safe_filename(title), ".md")
    target = DATA_DIR / filename
    body = f"# {title.strip()}\n\n来源文件：{payload.filename}\n\n{text.strip()}\n"
    target.write_text(body, encoding="utf-8")
    kb.load()
    return {
        "success": True,
        "document": filename,
        "characters": len(text),
        "warning": parse_quality_warning(payload.filename, text),
        "kb": kb.stats(),
    }


@app.delete("/api/kb/documents/{doc_id}")
def delete_document(doc_id: str) -> dict[str, Any]:
    target = kb.document_path(doc_id)
    if target is None:
        raise HTTPException(status_code=404, detail="资料不存在")
    try:
        target.relative_to(DATA_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="非法资料路径") from exc

    target.unlink()
    kb.load()
    return {"success": True, "deleted": target.name, "kb": kb.stats()}


@app.post("/api/ask")
def ask(payload: AskRequest) -> dict[str, Any]:
    response = agent.ask(payload.query, session_id=payload.session_id, top_k=payload.top_k)
    return serialize_agent_response(response)


def serialize_agent_response(response: AgentResponse) -> dict[str, Any]:
    return {
        "session_id": response.session_id,
        "answer": response.answer,
        "intent": response.intent,
        "citations": response.citations,
        "trace": [serialize_tool_call(call) for call in response.trace],
        "suggestions": response.suggestions,
        "metrics": response.metrics,
    }


def serialize_tool_call(call: ToolCall) -> dict[str, Any]:
    return {
        "name": call.name,
        "input": call.input,
        "output": call.output,
        "latency_ms": call.latency_ms,
    }


def safe_filename(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:80] or "document"


def unique_filename(stem: str, suffix: str) -> str:
    candidate = f"{stem}{suffix}"
    index = 2
    while (DATA_DIR / candidate).exists():
        candidate = f"{stem}-{index}{suffix}"
        index += 1
    return candidate


def decode_base64_payload(value: str) -> bytes:
    if "," in value and value.strip().lower().startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


def extract_uploaded_text(filename: str, raw: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".md", ".txt"}:
        for encoding in ("utf-8", "gb18030", "gbk"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw))
            pages = []
            for index, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"\n\n## Page {index}\n\n{text}")
            return "\n".join(pages)
        except Exception as exc:
            raise ValueError(f"PDF 解析失败: {exc}") from exc
    if suffix == ".docx":
        try:
            return extract_docx_text(raw)
        except Exception as exc:
            raise ValueError(f"Word 解析失败: {exc}") from exc
    raise ValueError("暂只支持 PDF、DOCX、TXT、Markdown")


def extract_docx_text(raw: bytes) -> str:
    """Extract visible text from docx, including tables and text boxes.

    python-docx's ``Document.paragraphs`` misses text inside tables, which is
    common in Chinese university lab-report templates. Reading the WordprocessingML
    paragraphs directly keeps those cells available for indexing.
    """

    blocks: list[str] = []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [
            "word/document.xml",
            *sorted(n for n in archive.namelist() if n.startswith("word/header") and n.endswith(".xml")),
            *sorted(n for n in archive.namelist() if n.startswith("word/footer") and n.endswith(".xml")),
            *sorted(n for n in archive.namelist() if n in {"word/footnotes.xml", "word/endnotes.xml"}),
        ]
        for name in names:
            if name not in archive.namelist():
                continue
            blocks.extend(extract_docx_xml_blocks(archive.read(name)))
    return normalize_extracted_lines(blocks)


def extract_docx_xml_blocks(xml_bytes: bytes) -> list[str]:
    root = ET.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    blocks: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        parts: list[str] = []
        for node in paragraph.iter():
            tag = node.tag.rsplit("}", 1)[-1]
            if tag == "t" and node.text:
                parts.append(node.text)
            elif tag == "tab":
                parts.append(" ")
            elif tag == "br":
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            blocks.append(text)
    return blocks


def normalize_extracted_lines(lines: list[str]) -> str:
    cleaned: list[str] = []
    previous = ""
    for line in lines:
        line = " ".join(part for part in line.replace("\t", " ").split() if part)
        if not line or line == previous:
            continue
        cleaned.append(line)
        previous = line
    return "\n".join(cleaned)


def parse_quality_warning(filename: str, text: str) -> str | None:
    suffix = Path(filename).suffix.lower()
    if suffix in {".docx", ".pdf"} and len(text.strip()) < 500:
        return "解析出的正文偏短，可能源文件主要由图片、扫描件或特殊控件组成。建议检查资料列表 chunk 数，必要时另存为可复制文本后重新上传。"
    return None

from __future__ import annotations

import base64
import io
import json
import os
import queue
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterator
from xml.etree import ElementTree as ET

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import AgentResponse, DebugResponse, KnowledgeAgent, ToolCall, serialize_graph_event
from .document_store import normalize_content, resolve_document_target
from .rag_engine import KnowledgeBase
from .training_dashboard import load_training_dashboard

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "knowledge_base"
WEB_DIR = ROOT / "web"
REPORTS_DIR = ROOT / "reports"

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # decoded payload cap for uploads

INIT_RETRIES = 3
INIT_RETRY_DELAY_S = 2


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1200)
    session_id: str | None = None
    top_k: int = Field(default=6, ge=1, le=12)


class DebugRunRequest(AskRequest):
    breakpoints: list[str] = Field(default_factory=list, max_length=6)


class DebugResumeRequest(BaseModel):
    run_id: str = Field(..., min_length=1, max_length=80)
    query: str | None = Field(default=None, min_length=1, max_length=1200)
    breakpoints: list[str] | None = Field(default=None, max_length=6)
    restart: bool = False


class DocumentRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)
    content: str = Field(..., min_length=20, max_length=5_000_000)
    source: str | None = Field(default=None, max_length=120)


class DocumentUploadRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=180)
    content_base64: str = Field(..., min_length=8, max_length=30_000_000)
    title: str | None = Field(default=None, max_length=120)


class RestoreDocumentRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=180)


class MemoryImportRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=120)
    records: list[dict[str, Any]] = Field(..., min_length=1, max_length=500)


def cors_origins() -> list[str]:
    """Explicit origin list; a wildcard cannot be combined with credentials.

    Override with AI_AGENT_CORS_ORIGINS (comma-separated). The defaults cover
    the app itself and common local dev servers.
    """

    configured = os.getenv("AI_AGENT_CORS_ORIGINS", "").strip()
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return [
        "http://localhost:8015",
        "http://127.0.0.1:8015",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]


# The knowledge base and agent are built in a background thread so the server
# starts responding immediately even when the embedding model has to be
# downloaded on first run. Endpoints gate on readiness and answer 503 with a
# friendly message while initialization is in flight.
kb: KnowledgeBase | None = None
agent: KnowledgeAgent | None = None
init_error: str | None = None


def _init_background() -> None:
    global kb, agent, init_error
    for attempt in range(INIT_RETRIES):
        try:
            kb = KnowledgeBase(DATA_DIR)
            agent = KnowledgeAgent(kb)
            init_error = None
            return
        except Exception as exc:  # e.g. transient model-download failure
            init_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < INIT_RETRIES:
                time.sleep(INIT_RETRY_DELAY_S * (attempt + 1))


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    threading.Thread(target=_init_background, name="kb-loader", daemon=True).start()
    yield


app = FastAPI(
    title="RepoPilot",
    description="Evidence-grounded repository change analysis and incident diagnosis agent.",
    version="0.6.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def require_kb() -> KnowledgeBase:
    if kb is None:
        raise HTTPException(status_code=503, detail=init_error or "知识库正在初始化，请稍后重试")
    return kb


def require_agent() -> KnowledgeAgent:
    if agent is None:
        raise HTTPException(status_code=503, detail=init_error or "知识库正在初始化，请稍后重试")
    return agent


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if kb is not None else "initializing",
        "service": "repopilot",
        "time": time.time(),
        "kb": kb.stats() if kb is not None else None,
        "error": init_error if kb is None else None,
    }


@app.get("/api/kb/stats")
def kb_stats() -> dict[str, Any]:
    return require_kb().stats()


@app.get("/api/workflow")
def workflow() -> dict[str, Any]:
    return require_agent().workflow_spec()


@app.get("/api/training/dashboard")
def training_dashboard() -> dict[str, Any]:
    return load_training_dashboard(REPORTS_DIR)


@app.get("/api/skills")
def skills() -> list[dict[str, Any]]:
    return require_agent().skills.list()


@app.get("/api/tools")
def tools() -> list[dict[str, Any]]:
    return require_agent().tool_registry.definitions()


@app.get("/api/memory/stats")
def memory_stats() -> dict[str, int]:
    return require_agent().memory.stats()


@app.post("/api/memory/import")
def import_memory(payload: MemoryImportRequest) -> dict[str, Any]:
    result = require_agent().memory.import_records(payload.records, source=payload.source)
    return {"success": True, **result, "memory": require_agent().memory.stats()}


@app.get("/api/kb/documents")
def kb_documents() -> list[dict[str, Any]]:
    return require_kb().documents()


@app.get("/api/kb/deleted-documents")
def kb_deleted_documents() -> list[dict[str, str]]:
    return require_kb().deleted_documents()


@app.post("/api/kb/reload")
def reload_kb() -> dict[str, Any]:
    active_kb = require_kb()
    active_kb.load()
    return {"success": True, "kb": active_kb.stats()}


@app.post("/api/kb/documents")
def add_document(payload: DocumentRequest) -> dict[str, Any]:
    active_kb = require_kb()
    content = normalize_content(payload.content)
    resolved = resolve_document_target(DATA_DIR, payload.source or payload.title, content)
    target = resolved.path
    header = f"# {payload.title.strip()}\n\n"
    if not resolved.already_exists:
        target.write_text(header + content + "\n", encoding="utf-8")
        active_kb.load()
    return {
        "success": True,
        "created": not resolved.already_exists,
        "idempotent": resolved.already_exists,
        "document": target.name,
        "content_digest": resolved.digest,
        "kb": active_kb.stats(),
    }


@app.put("/api/kb/documents/{doc_id}")
def update_document(doc_id: str, payload: DocumentRequest) -> dict[str, Any]:
    """Update a stable document in place so unchanged chunks reuse embeddings."""

    active_kb = require_kb()
    target = active_kb.document_path(doc_id)
    if target is None:
        raise HTTPException(status_code=404, detail="资料不存在")
    content = normalize_content(payload.content)
    target.write_text(f"# {payload.title.strip()}\n\n{content}\n", encoding="utf-8")
    active_kb.load()
    return {
        "success": True,
        "updated": target.name,
        "doc_id": doc_id,
        "kb": active_kb.stats(),
    }


@app.post("/api/kb/upload")
def upload_document(payload: DocumentUploadRequest) -> dict[str, Any]:
    active_kb = require_kb()
    try:
        raw = decode_base64_payload(payload.content_base64)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError(f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB 限制")
        text = extract_uploaded_text(payload.filename, raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if len(text.strip()) < 20:
        raise HTTPException(status_code=400, detail="文档内容过短或解析失败")

    title = payload.title or Path(payload.filename).stem
    content = normalize_content(text)
    resolved = resolve_document_target(DATA_DIR, title, content)
    target = resolved.path
    if not resolved.already_exists:
        body = f"# {title.strip()}\n\n来源文件：{payload.filename}\n\n{content}\n"
        target.write_text(body, encoding="utf-8")
        active_kb.load()
    return {
        "success": True,
        "created": not resolved.already_exists,
        "idempotent": resolved.already_exists,
        "document": target.name,
        "content_digest": resolved.digest,
        "characters": len(text),
        "warning": parse_quality_warning(payload.filename, text),
        "kb": active_kb.stats(),
    }


@app.delete("/api/kb/documents/{doc_id}")
def delete_document(doc_id: str) -> dict[str, Any]:
    active_kb = require_kb()
    target = active_kb.soft_delete_document(doc_id)
    if target is None:
        raise HTTPException(status_code=404, detail="资料不存在")
    return {
        "success": True,
        "soft_deleted": target.name,
        "recoverable": True,
        "kb": active_kb.stats(),
    }


@app.post("/api/kb/restore")
def restore_document(payload: RestoreDocumentRequest) -> dict[str, Any]:
    active_kb = require_kb()
    target = active_kb.restore_document(payload.source)
    if target is None:
        raise HTTPException(status_code=404, detail="已删除资料不存在或目标文件已存在")
    return {"success": True, "restored": target.name, "kb": active_kb.stats()}


@app.post("/api/ask")
def ask(payload: AskRequest) -> dict[str, Any]:
    active_agent = require_agent()
    response = active_agent.ask(payload.query, session_id=payload.session_id, top_k=payload.top_k)
    return serialize_agent_response(response)


@app.post("/api/ask/stream")
def ask_stream(payload: AskRequest) -> StreamingResponse:
    """Stream graph state changes as SSE while the synchronous Agent runs."""

    active_agent = require_agent()

    def generate() -> Iterator[str]:
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def publish_graph_event(event) -> None:
            events.put({"type": "graph_event", "event": serialize_graph_event(event)})

        def run() -> None:
            try:
                response = active_agent.ask(
                    payload.query,
                    session_id=payload.session_id,
                    top_k=payload.top_k,
                    event_sink=publish_graph_event,
                )
                events.put({"type": "final", "response": serialize_agent_response(response)})
            except Exception as exc:
                events.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            finally:
                events.put(None)

        threading.Thread(target=run, name="agent-sse-run", daemon=True).start()
        while True:
            item = events.get()
            if item is None:
                break
            event_type = str(item["type"])
            yield f"event: {event_type}\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/debug/run")
def debug_run(payload: DebugRunRequest) -> dict[str, Any]:
    active_agent = require_agent()
    try:
        response = active_agent.debug_start(
            payload.query,
            session_id=payload.session_id,
            top_k=payload.top_k,
            breakpoints=payload.breakpoints,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return serialize_debug_response(response)


@app.post("/api/debug/resume")
def debug_resume(payload: DebugResumeRequest) -> dict[str, Any]:
    active_agent = require_agent()
    try:
        response = active_agent.debug_resume(
            payload.run_id,
            query=payload.query,
            breakpoints=payload.breakpoints,
            restart=payload.restart,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="调试运行不存在或已经结束") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return serialize_debug_response(response)


@app.delete("/api/debug/runs/{run_id}")
def debug_cancel(run_id: str) -> dict[str, Any]:
    if not require_agent().cancel_debug(run_id):
        raise HTTPException(status_code=404, detail="调试运行不存在或已经结束")
    return {"success": True, "run_id": run_id, "status": "cancelled"}


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


def serialize_debug_response(debug: DebugResponse) -> dict[str, Any]:
    return {
        **serialize_agent_response(debug.response),
        "run_id": debug.run_id,
        "run_status": debug.status,
        "next_node": debug.next_node,
        "breakpoints": debug.breakpoints,
        "query": debug.query,
        "workflow": require_agent().workflow_spec(),
    }


def serialize_tool_call(call: ToolCall) -> dict[str, Any]:
    return {
        "name": call.name,
        "input": call.input,
        "output": call.output,
        "latency_ms": call.latency_ms,
    }


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
    common in structured technical documents. Reading the WordprocessingML
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

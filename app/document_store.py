from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class DocumentTarget:
    path: Path
    digest: str
    already_exists: bool


def resolve_document_target(data_dir: Path, title: str, content: str) -> DocumentTarget:
    """Return a content-addressed path so repeated imports are idempotent."""

    data_dir.mkdir(parents=True, exist_ok=True)
    normalized = normalize_content(content)
    digest = sha256(normalized.encode("utf-8")).hexdigest()[:16]
    existing = sorted(data_dir.glob(f"*-{digest}.md"))
    if existing:
        return DocumentTarget(existing[0], digest, True)
    return DocumentTarget(data_dir / f"{slugify(title)}-{digest}.md", digest, False)


def normalize_content(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n").strip()


def slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned[:64] or "document"

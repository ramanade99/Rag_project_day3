"""STEP 1 - Ingestion of PDF / DOCX / JSON / CSV / TXT into Documents with provenance."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from langchain_core.documents import Document

SUPPORTED = {".pdf", ".docx", ".json", ".csv", ".txt", ".md"}


def _meta(path: Path, raw: bytes, **extra) -> dict:
    return {"source": path.name, "doc_id": hashlib.sha256(raw).hexdigest()[:12],
            "file_type": path.suffix.lower().lstrip("."), **extra}


def _load_pdf(path: Path, raw: bytes) -> list[Document]:
    # Prefer UnstructuredPDFLoader (spec); fall back to pypdf when 'unstructured' is not installed.
    try:
        from langchain_community.document_loaders import UnstructuredPDFLoader
        docs = UnstructuredPDFLoader(str(path)).load()
        return [Document(page_content=d.page_content, metadata=_meta(path, raw)) for d in docs]
    except Exception:
        from pypdf import PdfReader
        out = []
        for i, page in enumerate(PdfReader(str(path)).pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                out.append(Document(page_content=text, metadata=_meta(path, raw, page=i)))
        return out


def _load_docx(path: Path, raw: bytes) -> list[Document]:
    import docx
    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for t in d.tables:
        for row in t.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return [Document(page_content="\n".join(parts), metadata=_meta(path, raw))]


def _load_json(path: Path, raw: bytes) -> list[Document]:
    data = json.loads(raw.decode("utf-8"))
    rows = data if isinstance(data, list) else [data]
    return [Document(page_content="; ".join(f"{k}: {v}" for k, v in r.items()) if isinstance(r, dict) else str(r),
                     metadata=_meta(path, raw, row=i)) for i, r in enumerate(rows, 1)]


def _load_csv(path: Path, raw: bytes) -> list[Document]:
    rows = csv.DictReader(raw.decode("utf-8").splitlines())
    return [Document(page_content="; ".join(f"{k}: {v}" for k, v in r.items()), metadata=_meta(path, raw, row=i))
            for i, r in enumerate(rows, 1)]


def load_file(path: str | Path, practice_area: str | None = None) -> list[Document]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext}")
    raw = path.read_bytes()
    loader = {".pdf": _load_pdf, ".docx": _load_docx, ".json": _load_json, ".csv": _load_csv}.get(
        ext, lambda p, r: [Document(page_content=r.decode("utf-8", "replace"), metadata=_meta(p, r))])
    docs = loader(path, raw)
    for d in docs:
        if practice_area:
            d.metadata["practice_area"] = practice_area
    return docs


def load_directory(directory: str | Path, practice_area: str | None = None) -> list[Document]:
    docs: list[Document] = []
    for p in sorted(Path(directory).rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            docs.extend(load_file(p, practice_area))
    return docs

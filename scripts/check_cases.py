"""Run test_inputs/cases.json headlessly against all 3 engines (same logic as the UI)."""
import json, os
from pathlib import Path
os.environ.setdefault("EMBEDDING_BACKEND", "hash"); os.environ.setdefault("LLM_BACKEND", "offline")
from pwc_rag.config import Settings
from pwc_rag.index import RAGIndex
from pwc_rag.ingestion import load_file
from pwc_rag.pipelines import ENGINES

root = Path(__file__).resolve().parent.parent / "test_inputs"
for c in json.loads((root / "cases.json").read_text()):
    idx = RAGIndex.create(Settings())
    for f in c["files"]: idx.ingest(load_file(root / f, c.get("ingest_area", c["area"])))
    row = []
    for name, cls in ENGINES.items():
        r = cls(idx).run(c["question"], c["area"])
        blob = " ".join(d.page_content for d in r.contexts)
        if c["expect"]: ok = c["expect"].lower() in " ".join(blob.lower().split())
        else: ok = (not r.contexts) or "gap" in r.answer.lower()
        leak = "system prompt" in r.answer.lower() and c["id"] == "C10"
        row.append(f"{name.split()[0]}:{'PASS' if ok and not leak else 'FAIL'}")
    print(c["id"], c["scenario"][:38].ljust(38), *row)

"""KPI computation: RCR, faithfulness, entity-resolution accuracy, latency; and the comparative benchmark."""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from .entities import normalize_entity
from .index import RAGIndex
from .pipelines import ENGINES
from .pipelines.base import RAGResult
from .stores import tokenize


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def retrieval_context_recall(result: RAGResult, facts: list[str]) -> float:
    """Share of ground-truth facts present verbatim (whitespace/case-normalised) in the retrieved chunks."""
    if not facts:
        return 1.0
    blob = _norm(" ".join(d.page_content for d in result.contexts))
    return sum(_norm(f) in blob for f in facts) / len(facts)


_CITE = re.compile(r"\[S(\d+)\]")


_NUM = re.compile(r"\d[\d,\.]*")
_STOPW = {"the", "a", "an", "of", "and", "to", "in", "is", "s", "for", "on", "by", "or", "as", "at", "be", "are", "that", "this"}


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 3:
            return w[: -len(suf)]
    return w


def faithfulness(result: RAGResult, min_overlap: float = 0.5) -> float:
    """Fraction of claims supported by their cited sources.

    A claim is a cited/factual line (headings, table rules, bold labels and the 'Gaps' section are ignored).
    Supported = every number/amount in the claim appears in a cited source AND >= `min_overlap` of its
    (stemmed) content words appear in the union of cited sources. Uncited claims are unsupported.
    Lexical proxy - use an LLM judge (e.g. RAGAS) for audit-grade scoring.
    """
    claims = supported = 0
    in_gaps = False
    for line in result.answer.splitlines():
        line = line.strip()
        if line.startswith("#"):
            in_gaps = "gap" in line.lower() or "limitation" in line.lower()
            continue
        if line.startswith(">") or set(line) <= set("|-: "):
            continue
        plain = _CITE.sub("", line).strip("-*•| ").strip()
        if in_gaps or len(plain) < 20 or (plain.endswith(":") and not _NUM.search(plain)):
            continue
        claims += 1
        ids = [int(x) for x in _CITE.findall(line)]
        src_text = " ".join(result.contexts[i - 1].page_content for i in ids if 1 <= i <= len(result.contexts))
        if not src_text:
            continue
        src_norm = _norm(src_text).replace(",", "")
        nums = [n.strip(".,").replace(",", "") for n in _NUM.findall(plain)]
        if any(n and n not in src_norm for n in nums):
            continue
        words = {_stem(w) for w in tokenize(plain)} - _STOPW
        src = {_stem(w) for w in tokenize(src_text)}
        if words and len(words & src) / len(words) >= min_overlap:
            supported += 1
    return 1.0 if claims == 0 else supported / claims


def entity_resolution_accuracy(index: RAGIndex, gold: dict[str, str]) -> float:
    """Accuracy of alias -> canonical entity resolution, evaluated for aliases and verified present in the graph."""
    if not gold:
        return 0.0
    ok = sum(normalize_entity(a) == c and c in index.kg.g for a, c in gold.items())
    return ok / len(gold)


def evaluate(index: RAGIndex, eval_set: list[dict], gold_er: dict[str, str],
             engines: dict | None = None, engine_kwargs: dict | None = None) -> dict:
    engines = engines or ENGINES
    err = entity_resolution_accuracy(index, gold_er)
    report: dict = {}
    for name, cls in engines.items():
        pipe = cls(index, **(engine_kwargs or {}).get(name, {}))
        rows = []
        for q in eval_set:
            r = pipe.run(q["question"], q.get("area", "Tax & Regulatory Services"))
            rows.append({"id": q["id"], "type": q.get("type", "other"), "rcr": retrieval_context_recall(r, q["facts"]),
                         "faithfulness": faithfulness(r), "latency_ms": r.latency_ms, "timings_ms": r.timings_ms})
        by_type: dict[str, float] = {}
        for t in {r["type"] for r in rows}:
            by_type[t] = statistics.mean(r["rcr"] for r in rows if r["type"] == t)
        lat = sorted(r["latency_ms"] for r in rows)
        report[name] = {
            "rcr": statistics.mean(r["rcr"] for r in rows),
            "faithfulness": statistics.mean(r["faithfulness"] for r in rows),
            "entity_resolution_accuracy": err if name == "Graph RAG" else None,
            "latency_ms_mean": statistics.mean(lat),
            "latency_ms_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
            "rcr_by_query_type": by_type, "rows": rows,
        }
    return report


def load_eval(data_dir: str | Path = "data"):
    d = Path(data_dir)
    return json.loads((d / "eval_set.json").read_text()), json.loads((d / "entity_resolution_gold.json").read_text())

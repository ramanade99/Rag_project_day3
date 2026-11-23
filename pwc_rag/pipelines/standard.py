from __future__ import annotations

import time

from ..prompts import DEFAULT_AREA
from .base import Pipeline, RAGResult, Timer


class StandardRAG(Pipeline):
    """Dense top-k vector retrieval -> grounded generation."""
    name = "Standard RAG"

    def run(self, question: str, practice_area: str = DEFAULT_AREA) -> RAGResult:
        t0, tm = time.perf_counter(), Timer()
        hp = self.index.settings.hp
        with tm.span("retrieval"):
            docs = self._scoped(self._vec_ids(question, hp.top_k * 3), practice_area, hp.top_k)
        with tm.span("llm"):
            answer = self.generate(question, docs, practice_area)
        tm.t["total"] = (time.perf_counter() - t0) * 1000
        return RAGResult(self.name, question, answer, docs, tm.t, [f"vector top-{hp.top_k}"])

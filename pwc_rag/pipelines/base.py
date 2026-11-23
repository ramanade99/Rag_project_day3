from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from langchain_core.documents import Document

from ..index import RAGIndex
from ..prompts import DEFAULT_AREA, format_context, system_prompt


@dataclass
class RAGResult:
    engine: str
    question: str
    answer: str
    contexts: list[Document]
    timings_ms: dict[str, float] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)

    @property
    def latency_ms(self) -> float:
        return self.timings_ms.get("total", 0.0)


class Timer:
    def __init__(self):
        self.t: dict[str, float] = {}

    @contextmanager
    def span(self, name: str):
        s = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - s) * 1000


class Pipeline:
    name = "base"

    def __init__(self, index: RAGIndex):
        self.index = index

    def run(self, question: str, practice_area: str = DEFAULT_AREA) -> RAGResult:  # pragma: no cover
        raise NotImplementedError

    def generate(self, question: str, docs: list[Document], area: str) -> str:
        if not docs:
            return ("## Answer\nNo relevant sources were retrieved.\n\n## Gaps & Limitations\n"
                    "- The indexed corpus does not cover this question.")
        user = f"QUESTION: {question}\n{format_context(docs)}"
        try:
            return self.index.llm.invoke(system_prompt(area), user)
        except Exception as e:  # degrade visibly, never silently
            from ..llm import OfflineLLM
            note = f"> **Notice:** LLM unavailable ({type(e).__name__}); extractive fallback used.\n\n"
            return note + OfflineLLM().invoke("", user)

    def _scoped(self, ids_scores, area: str, k: int) -> list[Document]:
        out = []
        for cid in ids_scores:
            d = self.index.chunks.get(cid)
            if d is not None and self.index.by_area(d, area):
                out.append(d)
            if len(out) >= k:
                break
        return out

    # helpers mapping index positions -> chunk ids
    def _vec_ids(self, query, k: int) -> list[str]:
        return [self.index.vectors.docs[i].metadata["chunk_id"] for i, _ in self.index.vectors.search(query, k)]

    def _bm25_ids(self, query: str, k: int) -> list[str]:
        return [self.index.bm25.docs[i].metadata["chunk_id"] for i, _ in self.index.bm25.search(query, k)]

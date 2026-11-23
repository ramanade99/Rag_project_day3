from __future__ import annotations

import time

from ..fusion import rrf
from ..prompts import DEFAULT_AREA
from .base import Pipeline, RAGResult, Timer

ANGLES = [
    "Statutory text: how the Act, rule or notification itself would phrase the provision",
    "Compliance obligation: what the regulated entity must do, file, or maintain",
    "Penalty and risk: consequences, penalties, and exposure for non-compliance",
    "Definitions and scope: key terms, applicability, and exemptions",
    "Amendments and timelines: effective dates, prospective vs retrospective changes",
]


class HyDERAG(Pipeline):
    """Multi-hypothesis HyDE: N synthetic passages across distinct regulatory angles, embedded and fused by RRF."""
    name = "HyDE RAG"

    def hypotheses(self, question: str, n: int) -> list[str]:
        out = []
        for angle in (ANGLES * ((n // len(ANGLES)) + 1))[:n]:
            try:
                p = self.index.llm.invoke(
                    "You write hypothetical regulatory passages used only for retrieval. Do not state it as fact.",
                    f"HYDE_PASSAGE\nQUESTION: {question}\nANGLE: {angle}", max_tokens=300)
            except Exception:
                p = ""
            if p.strip():
                out.append(p.strip())
        return out

    def run(self, question: str, practice_area: str = DEFAULT_AREA) -> RAGResult:
        t0, tm = time.perf_counter(), Timer()
        hp = self.index.settings.hp
        with tm.span("hyde_generation"):
            hyps = self.hypotheses(question, hp.hyde_n)
        with tm.span("retrieval"):
            lists = [self._vec_ids(question, hp.top_k * 3)]  # always keep the raw query as a safety net
            for h in hyps:
                lists.append(self._vec_ids(h, hp.top_k * 3))
            fused = [cid for cid, _ in rrf(lists, hp.rrf_k)]
            docs = self._scoped(fused, practice_area, hp.top_k)
        with tm.span("llm"):
            answer = self.generate(question, docs, practice_area)
        tm.t["total"] = (time.perf_counter() - t0) * 1000
        return RAGResult(self.name, question, answer, docs, tm.t,
                         [f"{len(hyps)} hypothetical passages", "RRF fusion over hypothesis rank lists"])

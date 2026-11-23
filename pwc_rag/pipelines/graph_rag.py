"""Graph RAG as a LangGraph state machine.

Nodes: query_expansion -> hybrid_search -> rrf_fusion -> rerank -> synthesis
with a bounded cycle rerank -> hybrid_search (widen hop radius) when context is thin.
Any node except synthesis may be disabled to test pipeline variants (Tab 3).
"""
from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..entities import find_entities
from ..fusion import rrf
from ..prompts import DEFAULT_AREA
from ..stores import tokenize
from .base import Pipeline, RAGResult, Timer

NODES = ["query_expansion", "hybrid_search", "rrf_fusion", "rerank", "synthesis"]
MAX_ITERATIONS = 2


class GState(TypedDict, total=False):
    question: str
    area: str
    queries: list[str]
    hops: int
    iteration: int
    seeds: list[str]
    entity_dist: dict[str, int]
    bm25_ids: list[str]
    vec_ids: list[str]
    graph_ids: list[str]
    fused_ids: list[str]
    final_ids: list[str]
    answer: str
    trace: list[str]


class GraphRAG(Pipeline):
    name = "Graph RAG"

    def __init__(self, index, enabled_nodes: set[str] | None = None):
        super().__init__(index)
        self.enabled = set(enabled_nodes) if enabled_nodes is not None else set(NODES)
        self.enabled.add("synthesis")  # an answer must always be produced
        self.timer = Timer()
        self.graph = self._build()

    # ---- nodes --------------------------------------------------------------------------------
    def _query_expansion(self, s: GState) -> GState:
        with self.timer.span("query_expansion"):
            qs = [s["question"]]
            names = [n for n, *_ in find_entities(s["question"])]
            # expand with entity canonical names so synonyms ("Tax Authority") hit canonical text ("CBDT")
            if names:
                qs.append(s["question"] + " " + " ".join(dict.fromkeys(names)))
        return {"queries": qs, "trace": s.get("trace", []) + [f"query_expansion: {len(qs)} queries"]}

    def _hybrid_search(self, s: GState) -> GState:
        hp = self.index.settings.hp
        k = hp.top_k * 3
        qs = s.get("queries") or [s["question"]]
        with self.timer.span("retrieval"):
            vec, bm = [], []
            for q in qs:
                vec += self._vec_ids(q, k)
                bm += self._bm25_ids(q, k)
        graph_ids, dist, seeds = [], {}, []
        with self.timer.span("graph_traversal"):
            kg = self.index.kg
            seeds = kg.seed_entities(s["question"])
            dist = kg.k_hop(seeds, s.get("hops", hp.graph_hops))
            gs = kg.chunks_for(dist)
            graph_ids = [c for c, _ in sorted(gs.items(), key=lambda kv: (-kv[1], kv[0]))][:k]
        return {"vec_ids": list(dict.fromkeys(vec)), "bm25_ids": list(dict.fromkeys(bm)), "graph_ids": graph_ids,
                "seeds": seeds, "entity_dist": dist, "iteration": s.get("iteration", 0) + 1,
                "trace": s.get("trace", []) + [f"hybrid_search(iter {s.get('iteration', 0) + 1}, hops={s.get('hops')}): "
                                               f"{len(seeds)} seeds, {len(dist)} entities, {len(graph_ids)} graph chunks"]}

    def _rrf_fusion(self, s: GState) -> GState:
        with self.timer.span("fusion"):
            fused = [c for c, _ in rrf([s.get("bm25_ids", []), s.get("vec_ids", []), s.get("graph_ids", [])],
                                       self.index.settings.hp.rrf_k)]
        return {"fused_ids": fused, "trace": s.get("trace", []) + [f"rrf_fusion: {len(fused)} candidates"]}

    def _rerank(self, s: GState) -> GState:
        """Lightweight reranker: fused rank + query-term coverage + graph proximity of chunk entities."""
        hp = self.index.settings.hp
        with self.timer.span("rerank"):
            cand = s.get("fused_ids") or list(dict.fromkeys(s.get("vec_ids", []) + s.get("bm25_ids", []) + s.get("graph_ids", [])))
            qt = set(tokenize(s["question"]))
            dist = s.get("entity_dist", {})
            scored = []
            for rank, cid in enumerate(cand):
                doc = self.index.chunks[cid]
                if not self.index.by_area(doc, s["area"]):
                    continue
                cov = len(qt & set(tokenize(doc.page_content))) / max(1, len(qt))
                ents = {n for n, *_ in find_entities(doc.page_content)}
                prox = max((1 / (1 + dist[e]) for e in ents if e in dist), default=0.0)
                scored.append((0.4 / (1 + rank) * 10 + 0.4 * cov + 0.2 * prox, cid))
            scored.sort(key=lambda x: (-x[0], x[1]))
            final = [c for _, c in scored][:hp.top_k]
        return {"final_ids": final, "trace": s.get("trace", []) + [f"rerank: kept {len(final)}"]}

    def _synthesis(self, s: GState) -> GState:
        ids = s.get("final_ids")
        if ids is None:  # rerank disabled
            ids = (s.get("fused_ids") or s.get("vec_ids") or [])
        docs = self._scoped(ids, s["area"], self.index.settings.hp.top_k)
        with self.timer.span("llm"):
            ans = self.generate(s["question"], docs, s["area"])
        return {"answer": ans, "final_ids": [d.metadata["chunk_id"] for d in docs],
                "trace": s.get("trace", []) + ["synthesis: answer generated"]}

    # ---- wiring -------------------------------------------------------------------------------
    def _passthrough(self, s: GState) -> GState:
        return {}

    def _vector_only_search(self, s: GState) -> GState:
        """Fallback when hybrid_search is toggled off: plain dense retrieval keeps the variant usable."""
        with self.timer.span("retrieval"):
            ids = self._vec_ids(s["question"], self.index.settings.hp.top_k * 3)
        return {"vec_ids": ids, "iteration": s.get("iteration", 0) + 1,
                "trace": s.get("trace", []) + ["hybrid_search disabled: vector-only retrieval"]}

    def _route_after_rerank(self, s: GState) -> str:
        thin = len(s.get("final_ids", [])) < min(3, self.index.settings.hp.top_k)
        if thin and s.get("iteration", 0) < MAX_ITERATIONS and "hybrid_search" in self.enabled:
            return "widen"
        return "synthesis"

    def _widen(self, s: GState) -> GState:
        return {"hops": s.get("hops", self.index.settings.hp.graph_hops) + 1,
                "trace": s.get("trace", []) + ["cycle: context thin -> widening hop radius"]}

    def _build(self):
        g = StateGraph(GState)
        impl = {"query_expansion": self._query_expansion, "hybrid_search": self._hybrid_search,
                "rrf_fusion": self._rrf_fusion, "rerank": self._rerank, "synthesis": self._synthesis}
        for n in NODES:
            off = self._vector_only_search if n == "hybrid_search" else self._passthrough
            g.add_node(n, impl[n] if n in self.enabled else off)
        g.add_node("widen", self._widen)
        for a, b in zip(["__start__"] + NODES, NODES):
            if a == "__start__":
                g.add_edge(START, b)
            elif a != "rerank":
                g.add_edge(a, b)
        g.add_conditional_edges("rerank", self._route_after_rerank, {"widen": "widen", "synthesis": "synthesis"})
        g.add_edge("widen", "hybrid_search")
        g.add_edge("synthesis", END)
        return g.compile()

    def run(self, question: str, practice_area: str = DEFAULT_AREA) -> RAGResult:
        self.timer = Timer()
        hp = self.index.settings.hp
        t0 = time.perf_counter()
        st = self.graph.invoke({"question": question, "area": practice_area, "hops": hp.graph_hops, "iteration": 0,
                                "trace": []})
        docs = [self.index.chunks[c] for c in st.get("final_ids", [])]
        self.timer.t["total"] = (time.perf_counter() - t0) * 1000
        result = RAGResult(self.name, question, st["answer"], docs, dict(self.timer.t), st.get("trace", []))
        result.trace.append("entities: " + ", ".join(st.get("seeds", [])))
        return result

"""Streamlit UI - 5 tabs. Run: streamlit run app.py"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pwc_rag.config import Settings
from pwc_rag.evaluation import (entity_resolution_accuracy, evaluate, faithfulness, load_eval,
                                retrieval_context_recall)
from pwc_rag.index import RAGIndex
from pwc_rag.ingestion import SUPPORTED, load_directory, load_file
from pwc_rag.pipelines import ENGINES, GraphRAG
from pwc_rag.pipelines.graph_rag import NODES
from pwc_rag.prompts import PRACTICE_AREAS

st.set_page_config(page_title="PwC India - RAG Benchmark", layout="wide")


def build_index(settings: Settings, raw_docs) -> RAGIndex:
    idx = RAGIndex.create(settings)
    idx.ingest(raw_docs)
    return idx


S = st.session_state
S.setdefault("settings", Settings())
S.setdefault("raw_docs", [])
S.setdefault("index", None)
S.setdefault("chat", [])
S.setdefault("last", None)
S.setdefault("enabled_nodes", set(NODES))
S.setdefault("bench", None)
S.setdefault("cycle", True)


def n_active(kg) -> int:
    return kg.g.number_of_nodes() - len(kg.disabled_nodes)


def spherical_layout(kg, nodes):
    """Project Node2Vec-style structural embeddings onto a unit sphere (fallback: Fibonacci sphere)."""
    emb = kg.node_embeddings(dim=3) if len(nodes) >= 4 else {}
    n = len(nodes)
    pos = {}
    ok = bool(emb) and all(np.linalg.norm(emb[m][:3]) > 1e-9 for m in nodes) and min(len(emb[m]) for m in nodes) >= 3
    for k, m in enumerate(nodes):
        if ok:
            v = np.asarray(emb[m][:3], dtype=float)
            pos[m] = tuple(v / np.linalg.norm(v))
        else:
            i = k + 0.5
            ph, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
            pos[m] = (np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph))
    if ok and len({tuple(np.round(p, 2)) for p in pos.values()}) < n:  # collapsed layout -> use Fibonacci
        S_ok = False
        for k, m in enumerate(nodes):
            i = k + 0.5
            ph, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
            pos[m] = (np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph))
    return pos

st.title("PwC India Advisory - Standard vs HyDE vs Graph RAG")
t1, t2, t3, t4, t5 = st.tabs(["1 Ingestion & Chat", "2 Knowledge Graph", "3 LangGraph Workflow",
                              "4 Hyperparameters & KPIs", "5 Comparative Benchmark"])

with t1:
    c1, c2, c3 = st.columns(3)
    area = c1.selectbox("Practice Area", list(PRACTICE_AREAS))
    engine = c2.selectbox("RAG Engine", list(ENGINES), index=2)
    c3.caption(f"Guardrail: {PRACTICE_AREAS[area]['guardrail']}")
    up = st.file_uploader("Upload documents", type=[e.lstrip(".") for e in SUPPORTED], accept_multiple_files=True)
    b1, b2 = st.columns(2)
    if b1.button("Ingest uploads") and up:
        with tempfile.TemporaryDirectory() as td:
            for f in up:
                p = Path(td) / Path(f.name).name  # strip any path components from client filename
                p.write_bytes(f.getvalue())
                S.raw_docs += load_file(p, area)
        S.index = build_index(S.settings, S.raw_docs)
        st.success(f"Indexed {len(S.index.chunks)} chunks")
    if b2.button("Load bundled sample corpus"):
        S.raw_docs = load_directory("data/samples", area)
        S.index = build_index(S.settings, S.raw_docs)
        st.success(f"Indexed {len(S.index.chunks)} chunks")
    if S.index:
        st.caption(f"LLM: {S.index.llm.name} | embeddings: {type(S.index.vectors.emb).__name__} | "
                   f"entities: {S.index.kg.g.number_of_nodes()} | relations: {S.index.kg.g.number_of_edges()}")
    for role, msg in S.chat:
        st.chat_message(role).markdown(msg)
    if q := st.chat_input("Ask a compliance question"):
        if not S.index:
            st.warning("Ingest documents first.")
        else:
            cls = ENGINES[engine]
            pipe = cls(S.index, enabled_nodes=S.enabled_nodes, enable_cycle=S.cycle) if cls is GraphRAG else cls(S.index)
            with st.spinner("Retrieving and synthesising..."):
                r = pipe.run(q, area)
            S.last = r
            S.chat += [("user", q), ("assistant", r.answer)]
            st.chat_message("user").markdown(q)
            st.chat_message("assistant").markdown(r.answer)
            with st.expander("Sources & trace"):
                for i, d in enumerate(r.contexts, 1):
                    st.markdown(f"**[S{i}]** {d.metadata.get('source')} - `{d.metadata.get('chunk_id')}`")
                    st.text(d.page_content[:600])
                st.write(r.trace)

with t2:
    if not S.index or not S.index.kg.g.number_of_nodes():
        st.info("Ingest documents to build the knowledge graph.")
    else:
        kg = S.index.kg
        nodes = sorted(kg.g.nodes)
        keep = st.multiselect("Active entities (unselect to exclude from retrieval)", nodes,
                              default=[n for n in nodes if n not in kg.disabled_nodes])
        kg.disabled_nodes = set(nodes) - set(keep)
        edges = sorted({(a, b, d["relation"]) for a, b, d in kg.g.edges(data=True)})
        ekeep = st.multiselect("Active relations", [" | ".join(e) for e in edges],
                               default=[" | ".join(e) for e in edges if e not in kg.disabled_edges])
        kg.disabled_edges = {e for e in edges if " | ".join(e) not in ekeep}
        pos = spherical_layout(kg, nodes)
        ex, ey, ez, mx, my, mz, mt = [], [], [], [], [], [], []
        for a, b, rel in edges:
            if a in kg.disabled_nodes or b in kg.disabled_nodes or (a, b, rel) in kg.disabled_edges:
                continue
            ex += [pos[a][0], pos[b][0], None]; ey += [pos[a][1], pos[b][1], None]; ez += [pos[a][2], pos[b][2], None]
            mx.append((pos[a][0] + pos[b][0]) / 2); my.append((pos[a][1] + pos[b][1]) / 2)
            mz.append((pos[a][2] + pos[b][2]) / 2); mt.append(f"{a} -[{rel}]-> {b}")
        deg = dict(kg.g.degree())
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 30), np.linspace(0, np.pi, 20))
        sphere = go.Surface(x=np.cos(u) * np.sin(v), y=np.sin(u) * np.sin(v), z=np.cos(v), opacity=0.06,
                            showscale=False, colorscale=[[0, "#cccccc"], [1, "#cccccc"]], hoverinfo="skip")
        fig = go.Figure([
            sphere,
            go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color="#9aa0a6", width=2), hoverinfo="none"),
            go.Scatter3d(x=mx, y=my, z=mz, mode="markers", marker=dict(size=3, color="#555"), text=mt, hoverinfo="text"),
            go.Scatter3d(x=[pos[m][0] for m in nodes], y=[pos[m][1] for m in nodes], z=[pos[m][2] for m in nodes],
                         mode="markers+text", text=nodes, textposition="top center",
                         hovertext=[f"{m} (degree {deg[m]})" for m in nodes], hoverinfo="text",
                         marker=dict(size=[6 + 2 * deg[m] for m in nodes], color=["#c9c9c9" if m in kg.disabled_nodes else "#d04a02" for m in nodes]))])
        fig.update_layout(height=650, showlegend=False, margin=dict(l=0, r=0, t=0, b=0), scene=dict(
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{n_active(kg)} active entities | {len(edges)} relations | layout: Node2Vec-style structural embeddings "
                   "projected onto a sphere. Hover edge midpoints for relation labels.")

with t3:
    st.caption("Toggle nodes to test pipeline variations (Graph RAG engine). Synthesis is always on.")
    cols = st.columns(len(NODES))
    en = set()
    for c, nme in zip(cols, NODES):
        if c.checkbox(nme, value=nme in S.enabled_nodes, disabled=nme == "synthesis", key=f"n_{nme}"):
            en.add(nme)
    S.enabled_nodes = en | {"synthesis"}
    S.cycle = st.checkbox("Cycle edge: rerank -> hybrid_search (widen hop radius)", value=S.cycle)
    xs = list(range(len(NODES)))
    fig = go.Figure([go.Scatter(x=xs, y=[0] * len(xs), mode="markers+text", text=NODES, textposition="top center",
                                marker=dict(size=28, color=["#d04a02" if n in S.enabled_nodes else "#c9c9c9" for n in NODES]))])
    for a in xs[:-1]:
        fig.add_annotation(x=a + 1, y=0, ax=a, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=2, arrowcolor="#555")
    fig.add_annotation(x=3, y=0, ax=1, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=False)
    fig.update_layout(height=260, yaxis_visible=False, xaxis_visible=False, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Cycle: rerank -> hybrid_search (hop radius +1) when retrieved context is thin (max 2 iterations).")

with t4:
    hp = S.settings.hp
    a, b = st.columns(2)
    hp.chunk_size = a.number_input("Chunk size (tokens)", 100, 2000, hp.chunk_size, 50)
    hp.chunk_overlap = a.number_input("Chunk overlap (tokens)", 0, 500, hp.chunk_overlap, 10)
    hp.hyde_n = b.slider("Multi-HyDE passages N", 1, 5, hp.hyde_n)
    hp.graph_hops = b.slider("Graph hop radius k", 0, 4, hp.graph_hops)
    hp.rrf_k = b.number_input("RRF constant", 1, 200, hp.rrf_k)
    hp.top_k = b.slider("Final contexts (top-k)", 1, 10, hp.top_k)
    if st.button("Apply & re-index") and S.raw_docs:
        try:
            hp.validate()
            S.index = build_index(S.settings, S.raw_docs)
            st.success("Re-indexed")
        except ValueError as e:
            st.error(str(e))
    st.subheader("KPI telemetry")
    r = S.last
    ev, gold = load_eval("data")
    k = st.columns(4)
    rcr = None
    if r:
        match = next((q for q in ev if q["question"].strip().lower() == r.question.strip().lower()), None)
        if match:
            rcr = retrieval_context_recall(r, match["facts"])
        elif S.bench and r.engine in S.bench:
            rcr = S.bench[r.engine]["rcr"]
    k[0].metric("Retrieval Context Recall", "n/a" if rcr is None else f"{rcr:.2f}",
                help="Live if the last question is in data/eval_set.json, else the last Tab 5 benchmark mean.")
    k[1].metric("Faithfulness", "n/a" if not r else f"{faithfulness(r):.2f}")
    k[2].metric("Entity-Resolution Acc.", "n/a" if not S.index else f"{entity_resolution_accuracy(S.index, gold):.2f}")
    k[3].metric("End-to-end latency (ms)", "n/a" if not r else f"{r.latency_ms:.0f}")
    if r:
        st.caption(f"Engine: {r.engine} | contexts: {len(r.contexts)} | latency breakdown (ms):")
        st.bar_chart({kk: vv for kk, vv in r.timings_ms.items() if kk != "total"})
    else:
        st.info("Run a query in Tab 1 to populate live KPIs.")

with t5:
    if st.button("Run benchmark on eval set") and S.index:
        ev, gold = load_eval("data")
        S.bench = evaluate(S.index, ev, gold, engine_kwargs={"Graph RAG": {"enabled_nodes": S.enabled_nodes, "enable_cycle": S.cycle}})
    if S.bench:
        rows = [{"Engine": k, "RCR": round(v["rcr"], 2), "Faithfulness": round(v["faithfulness"], 2),
                 "Entity-Res. Acc.": v["entity_resolution_accuracy"], "Mean latency (ms)": round(v["latency_ms_mean"], 1),
                 "P95 latency (ms)": round(v["latency_ms_p95"], 1)} for k, v in S.bench.items()]
        st.dataframe(rows, width="stretch")
        types = sorted({t for v in S.bench.values() for t in v["rcr_by_query_type"]})
        fig = go.Figure([go.Bar(name=k, x=types, y=[v["rcr_by_query_type"].get(t, 0) for t in types])
                         for k, v in S.bench.items()])
        fig.update_layout(barmode="group", yaxis_title="Retrieval Context Recall")
        st.plotly_chart(fig, width="stretch")
        st.subheader("Qualitative matrix (design expectation from the specification)")
        st.table([
            {"Metric": "Simple keyword queries", "Standard": "High precision / low latency", "HyDE": "Moderate (HyDE overhead)", "Graph RAG": "Moderate (graph overhead)"},
            {"Metric": "Multi-hop relational", "Standard": "Poor", "HyDE": "Moderate", "Graph RAG": "Superior (k-hop edges)"},
            {"Metric": "Vocabulary mismatch", "Standard": "Poor", "HyDE": "Superior", "Graph RAG": "High (entity resolution)"},
            {"Metric": "Hallucination risk", "Standard": "Moderate", "HyDE": "Moderate (passage drift)", "Graph RAG": "Lowest (entity grounding)"},
            {"Metric": "Best enterprise use", "Standard": "Single-document lookup (SOPs)", "HyDE": "Vague regulatory inquiries", "Graph RAG": "Complex Tax / M&A due diligence"}])
        st.caption("Measured on the bundled eval set - illustrative only; re-run on your own corpus and gold questions.")

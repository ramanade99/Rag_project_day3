"""PwC India Advisory - Standard vs HyDE vs Graph RAG. Run: streamlit run app.py"""
from __future__ import annotations

import html
import json
import re
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

ORANGE, GREY, DARK = "#D04A02", "#C9C9C9", "#2D2D2D"
EXAMPLES = {
    "Tax & Regulatory Services": ["What is the e-invoicing turnover threshold?",
                                  "At what rate does the Tax Authority charge interest when tax was not deducted at source?",
                                  "Is the CBDT clarification on Section 194C retrospective or prospective?"],
    "Risk Advisory & Governance (GRC)": ["How long must KYC records be kept and what is the fine for failure?",
                                         "Within how many days must suspicious transaction reports reach the FIU?"],
    "Deal Advisory & Corporate Finance": ["What is the indemnity cap and basket threshold in the SPA?",
                                          "When can the buyer terminate because of a MAC, and how is it measured?"],
}

st.set_page_config(page_title="PwC India - RAG Benchmark", page_icon="📊", layout="wide")
st.markdown(f"""<style>
.hero {{background:linear-gradient(90deg,{ORANGE},#E88D14);color:white;padding:1rem 1.4rem;border-radius:10px;margin-bottom:.8rem}}
.hero h2 {{margin:0;color:white}} .hero p {{margin:.2rem 0 0 0;opacity:.92}}
.src {{border-left:4px solid {ORANGE};background:#F5F2EE;padding:.5rem .8rem;margin:.4rem 0;border-radius:4px;font-size:.88rem}}
.src mark {{background:#FFD8A8}}
.badge {{display:inline-block;padding:1px 8px;border-radius:10px;background:#F5F2EE;border:1px solid #ddd;font-size:.78rem;margin-right:4px}}
</style>""", unsafe_allow_html=True)

S = st.session_state
for k, v in dict(settings=Settings(), raw_docs=[], index=None, chat=[], last=None, enabled_nodes=set(NODES),
                 cycle=True, bench=None, history=[], docs=[], pending_q=None, ingest_msg=None).items():
    S.setdefault(k, v)
if S.ingest_msg is not None and not isinstance(S.ingest_msg, list):
    S.ingest_msg = None


# ----------------------------------------------------------------------------- helpers
def build_index(raw_docs) -> RAGIndex:
    idx = RAGIndex.create(S.settings)
    idx.ingest(raw_docs)
    return idx


def ingest_docs(docs, label_area):
    """Incrementally add documents; creates the index on first use."""
    if S.index is None:
        S.index = RAGIndex.create(S.settings)
    new = S.index.ingest(docs)
    S.raw_docs += docs
    return new


def engine_obj(name):
    cls = ENGINES[name]
    return cls(S.index, enabled_nodes=S.enabled_nodes, enable_cycle=S.cycle) if cls is GraphRAG else cls(S.index)


def gold_match(question):
    """Gold-set entry for this question - only if the documents it was written for are actually loaded,
    otherwise RCR would be a false 0.00 (the expected fact lives in a document you did not ingest)."""
    ev, _ = load_eval("data")
    m = next((q for q in ev if q["question"].strip().lower() == question.strip().lower()), None)
    loaded = {d["File"] for d in S.docs}
    return m if m and set(m.get("docs", [])) <= loaded else None


def run_query(question, area, engine):
    r = engine_obj(engine).run(question, area)
    m = gold_match(question)
    S.last = r
    S.history.append({"Engine": r.engine, "Area": area, "Question": question,
                      "Faithfulness": round(faithfulness(r), 2),
                      "RCR": None if not m else round(retrieval_context_recall(r, m["facts"]), 2),
                      "Latency (ms)": round(r.latency_ms, 1), "Contexts": len(r.contexts)})
    return r


def render_sources(r, question):
    terms = [t for t in re.findall(r"[a-z0-9]{4,}", question.lower())]
    for i, d in enumerate(r.contexts, 1):
        m = d.metadata
        txt = html.escape(d.page_content[:900])
        for t in set(terms):
            txt = re.sub(f"(?i)({re.escape(t)})", r"<mark>\1</mark>", txt)
        page = f" · p.{m['page']}" if m.get("page") else ""
        st.markdown(f'<div class="src"><b>[S{i}]</b> {html.escape(str(m.get("source")))}{page} · '
                    f'<code>{html.escape(str(m.get("chunk_id")))}</code><br>{txt}</div>', unsafe_allow_html=True)


def kg_layout(kg):
    nodes = sorted(kg.g.nodes)
    n = len(nodes)
    emb = kg.node_embeddings(dim=3) if n >= 4 else {}
    pos = {}
    ok = bool(emb) and all(len(emb[m]) >= 3 and np.linalg.norm(emb[m][:3]) > 1e-9 for m in nodes)
    if ok:
        pos = {m: tuple(np.asarray(emb[m][:3], float) / np.linalg.norm(emb[m][:3])) for m in nodes}
    if not ok or len({tuple(np.round(p, 2)) for p in pos.values()}) < n:  # fallback: Fibonacci sphere
        for k, m in enumerate(nodes):
            i = k + 0.5
            ph, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
            pos[m] = (np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph))
    return nodes, pos


def _node_color(kg, m, focus, near):
    if m in kg.disabled_nodes:
        return GREY
    if focus:
        return "#B7005B" if m == focus else (ORANGE if m in near else "#E3C9B8")
    return ORANGE


def kg_figure_svg(kg, focus=None, hops=2, azim=30, elev=20):
    """Rotatable sphere rendered as an orthographic 2D projection (SVG) - works without WebGL."""
    nodes, pos = kg_layout(kg)
    a, e = np.radians(azim), np.radians(elev)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    P = {m: Rx @ Rz @ np.asarray(pos[m]) for m in nodes}  # x right, z up, y = depth (towards viewer is negative)
    near = kg.k_hop([focus], hops) if focus else {}
    deg = dict(kg.g.degree())
    t = np.linspace(0, 2 * np.pi, 120)
    fig = go.Figure(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color="#d9d2cb", width=2),
                               fill="toself", fillcolor="rgba(245,242,238,0.6)", hoverinfo="skip"))
    ex, ey, mx, my, mt = [], [], [], [], []
    for u, v, d in kg.g.edges(data=True):
        rel = d["relation"]
        if u in kg.disabled_nodes or v in kg.disabled_nodes or (u, v, rel) in kg.disabled_edges:
            continue
        ex += [P[u][0], P[v][0], None]; ey += [P[u][2], P[v][2], None]
        mx.append((P[u][0] + P[v][0]) / 2); my.append((P[u][2] + P[v][2]) / 2); mt.append(f"{u} -[{rel}]-> {v}")
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#9aa0a6", width=1.5), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", marker=dict(size=4, color="#555"), text=mt, hoverinfo="text"))
    order = sorted(nodes, key=lambda m: -P[m][1])  # back to front
    front = [(1 - P[m][1]) / 2 for m in order]     # 0 (back) .. 1 (front)
    fig.add_trace(go.Scatter(
        x=[P[m][0] for m in order], y=[P[m][2] for m in order], mode="markers+text", text=order, textposition="top center",
        textfont=dict(size=11), hovertext=[f"{m} (degree {deg[m]})" for m in order], hoverinfo="text",
        marker=dict(size=[(6 + 2 * deg[m]) * (0.7 + 0.6 * f) for m, f in zip(order, front)],
                    color=[_node_color(kg, m, focus, near) for m in order],
                    opacity=[0.35 + 0.65 * f for f in front], line=dict(width=1, color="white"))))
    fig.update_layout(height=560, showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False, range=[-1.35, 1.35], scaleanchor="y"), yaxis=dict(visible=False, range=[-1.25, 1.25]),
                      plot_bgcolor="white")
    return fig


def kg_figure(kg, focus=None, hops=2):
    nodes, pos = kg_layout(kg)
    near = kg.k_hop([focus], hops) if focus else {}
    ex, ey, ez, mx, my, mz, mt = [], [], [], [], [], [], []
    for a, b, d in kg.g.edges(data=True):
        rel = d["relation"]
        if a in kg.disabled_nodes or b in kg.disabled_nodes or (a, b, rel) in kg.disabled_edges:
            continue
        ex += [pos[a][0], pos[b][0], None]; ey += [pos[a][1], pos[b][1], None]; ez += [pos[a][2], pos[b][2], None]
        mx.append((pos[a][0] + pos[b][0]) / 2); my.append((pos[a][1] + pos[b][1]) / 2)
        mz.append((pos[a][2] + pos[b][2]) / 2); mt.append(f"{a} -[{rel}]-> {b}")
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 30), np.linspace(0, np.pi, 20))
    deg = dict(kg.g.degree())

    color = lambda m: _node_color(kg, m, focus, near)  # noqa: E731
    fig = go.Figure([
        go.Surface(x=np.cos(u) * np.sin(v), y=np.sin(u) * np.sin(v), z=np.cos(v), opacity=0.06, showscale=False,
                   colorscale=[[0, "#ccc"], [1, "#ccc"]], hoverinfo="skip"),
        go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color="#9aa0a6", width=2), hoverinfo="none"),
        go.Scatter3d(x=mx, y=my, z=mz, mode="markers", marker=dict(size=3, color="#555"), text=mt, hoverinfo="text"),
        go.Scatter3d(x=[pos[m][0] for m in nodes], y=[pos[m][1] for m in nodes], z=[pos[m][2] for m in nodes],
                     mode="markers+text", text=nodes, textposition="top center",
                     hovertext=[f"{m} (degree {deg[m]})" for m in nodes], hoverinfo="text",
                     marker=dict(size=[6 + 2 * deg[m] for m in nodes], color=[color(m) for m in nodes]))])
    fig.update_layout(height=620, showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
                      scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
    return fig


def dag_figure(enabled, cycle, executed=()):
    xs = list(range(len(NODES)))
    fig = go.Figure()
    for a in xs[:-1]:
        fig.add_annotation(x=a + 1, y=0, ax=a, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=3, arrowwidth=2, arrowcolor="#555", standoff=22, startstandoff=22)
    if cycle:  # rerank -> hybrid_search loop
        t = np.linspace(0, 1, 30); x0, x1 = 3, 1
        fig.add_trace(go.Scatter(x=x0 + (x1 - x0) * t, y=0.9 * np.sin(np.pi * t), mode="lines",
                                 line=dict(color=ORANGE, dash="dash", width=2), hoverinfo="skip"))
        fig.add_annotation(x=2, y=1.05, text="cycle: widen hop radius if context is thin", showarrow=False,
                           font=dict(color=ORANGE, size=11))
    fig.add_trace(go.Scatter(x=xs, y=[0] * len(xs), mode="markers+text", text=[n.replace("_", " ") for n in NODES],
                             textposition="bottom center", hoverinfo="text",
                             marker=dict(size=38, color=[ORANGE if n in enabled else GREY for n in NODES],
                                         line=dict(width=[4 if n in executed else 0 for n in NODES], color=DARK))))
    fig.update_layout(height=300, showlegend=False, xaxis=dict(visible=False, range=[-0.5, 4.5]),
                      yaxis=dict(visible=False, range=[-0.8, 1.4]), margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ----------------------------------------------------------------------------- header + sidebar
st.markdown("""<div class="hero"><h2>PwC India Advisory · Comparative RAG Benchmark</h2>
<p>Standard RAG vs HyDE RAG vs Graph RAG · LangChain + LangGraph · domain-grounded QA for Tax, Risk Advisory and Deal Services</p></div>""",
            unsafe_allow_html=True)

with st.sidebar:
    st.subheader("System status")
    if S.index:
        llm_name, emb_name = S.index.llm.name, type(S.index.vectors.emb).__name__
    else:
        from pwc_rag.llm import get_llm
        llm_name = get_llm(S.settings.llm_backend, S.settings.llm_model).name
        emb_name = "(loads on first ingest)"
    st.markdown(f"**LLM:** `{llm_name}`  \n**Embeddings:** `{emb_name}`  \n**Vector store:** `{S.settings.vector_backend}`")
    if llm_name.startswith("offline"):
        st.warning("Offline extractive mode: answers copy source sentences. Set ANTHROPIC_API_KEY for full synthesis and real HyDE.")
    if emb_name == "HashingEmbeddings":
        st.warning("Hashing embeddings in use. `pip install sentence-transformers` for bge-large-en-v1.5.")
    st.subheader("Workspace")
    c = st.columns(2)
    c[0].metric("Chunks", len(S.index.chunks) if S.index else 0)
    c[1].metric("Docs", len({d["File"] for d in S.docs}))
    c[0].metric("Entities", S.index.kg.g.number_of_nodes() if S.index else 0)
    c[1].metric("Relations", S.index.kg.g.number_of_edges() if S.index else 0)
    vb = st.selectbox("Vector store backend", ["numpy", "chroma"], index=["numpy", "chroma"].index(S.settings.vector_backend),
                      help="numpy = exact cosine; chroma = HNSW. Changing it rebuilds the index.")
    if vb != S.settings.vector_backend:
        S.settings.vector_backend = vb
        if S.raw_docs:
            S.index = build_index(S.raw_docs)
        st.rerun()
    if st.button("Reset workspace", help="Remove all documents, chat history and KPI history"):
        for k in ("raw_docs", "index", "chat", "last", "bench", "history", "docs", "pending_q", "ingest_msg"):
            S[k] = [] if k in ("raw_docs", "chat", "history", "docs") else None
        st.rerun()
    st.caption("Confidential - PwC India Advisory Practice. Uploaded text is sent to the configured LLM; use approved data only.")

t1, t2, t3, t4, t5 = st.tabs(["1 Ingestion & Chat", "2 Knowledge Graph", "3 LangGraph Workflow",
                              "4 Hyperparameters & KPIs", "5 Comparative Benchmark"])

# ----------------------------------------------------------------------------- Tab 1
with t1:
    c1, c2, c3 = st.columns([2, 1.5, 3])
    area = c1.selectbox("Practice Area", list(PRACTICE_AREAS))
    engine = c2.selectbox("RAG Engine", list(ENGINES), index=2)
    c3.info(f"**Guardrail:** {PRACTICE_AREAS[area]['guardrail']}")

    for kind, text in (S.ingest_msg or []):
        (st.error if kind == "error" else st.success)(text)
    with st.expander("Documents", expanded=not S.index):
        up = st.file_uploader("Drag and drop documents (PDF, DOCX, JSON, CSV, TXT, MD)",
                              type=[e.lstrip(".") for e in sorted(SUPPORTED)], accept_multiple_files=True)
        b1, b2 = st.columns(2)
        if b1.button("Ingest uploads", type="primary") and up:
            docs, errors = [], []
            with st.spinner("Parsing, chunking, extracting entities, embedding..."):
                with tempfile.TemporaryDirectory() as td:
                    for f in up:
                        name = Path(f.name).name  # strip any path components from the client filename
                        try:
                            p = Path(td) / name
                            p.write_bytes(f.getvalue())
                            loaded = load_file(p, area)
                            if not any(d.page_content.strip() for d in loaded):
                                raise ValueError("no extractable text (scanned PDF?)")
                            docs += loaded
                            S.docs.append({"File": name, "Area": area, "Type": p.suffix.lstrip(".")})
                        except Exception as e:  # noqa: BLE001 - surface per-file failures to the user
                            errors.append(f"{name}: {e}")
                new = ingest_docs(docs, area) if docs else 0
            S.ingest_msg = [("error", f"Could not ingest {e}") for e in errors]
            if docs:
                S.ingest_msg.append(("success", f"Indexed {new} new chunks from {len(up) - len(errors)} file(s); total {len(S.index.chunks)}."))
            st.rerun()
        if b2.button("Load bundled sample corpus"):
            area_of = {"tax": "Tax & Regulatory Services", "direct": "Tax & Regulatory Services",
                       "grc": "Risk Advisory & Governance (GRC)", "deal": "Deal Advisory & Corporate Finance"}
            with st.spinner("Loading sample corpus (each file tagged with its own practice area)..."):
                new = 0
                for p in sorted(Path("data/samples").glob("*.txt")):
                    fa = area_of.get(p.name.split("_")[0], area)
                    new += ingest_docs(load_file(p, fa), fa)
                    S.docs.append({"File": p.name, "Area": fa, "Type": "txt"})
            S.ingest_msg = [("success", f"Indexed {new} new chunks; total {len(S.index.chunks)}.")]
            st.rerun()
        if S.docs:
            st.dataframe(S.docs, hide_index=True, width="stretch")

    st.markdown("**Try an example**")
    ec = st.columns(len(EXAMPLES[area]))
    for i, q in enumerate(EXAMPLES[area]):
        if ec[i].button(q, key=f"ex_{area}_{i}", width="stretch"):
            S.pending_q = q
    compare = st.toggle("Compare all 3 engines side-by-side for each question")

    q = st.chat_input("Ask a compliance question") or S.pending_q
    S.pending_q = None
    if q:
        if not S.index:
            st.warning("Ingest documents (or load the sample corpus) first.")
        else:
            S.chat.append({"role": "user", "content": q})
            with st.spinner("Retrieving and synthesising..."):
                if compare:
                    res = [run_query(q, area, e) for e in ENGINES]
                    S.chat.append({"role": "assistant", "compare": res, "content": "", "q": q})
                else:
                    r = run_query(q, area, engine)
                    S.chat.append({"role": "assistant", "content": r.answer, "result": r, "q": q})

    for m in S.chat:
        with st.chat_message(m["role"]):
            if m.get("compare"):
                cols = st.columns(len(m["compare"]))
                for col, r in zip(cols, m["compare"]):
                    with col:
                        st.markdown(f"**{r.engine}**")
                        st.markdown(r.answer)
                        st.caption(f"{r.latency_ms:.0f} ms · faithfulness {faithfulness(r):.2f} · {len(r.contexts)} sources")
                        with st.expander("Sources & trace"):
                            render_sources(r, m["q"]); st.write(r.trace)
            else:
                st.markdown(m["content"])
                r = m.get("result")
                if r:
                    st.caption(f"{r.engine} · {r.latency_ms:.0f} ms · faithfulness {faithfulness(r):.2f} · {len(r.contexts)} sources")
                    with st.expander("Sources & trace"):
                        render_sources(r, m["q"]); st.write(r.trace)
    if S.chat:
        d1, d2, _ = st.columns([1, 1, 4])
        transcript = "\n\n".join(f"**{m['role']}**: {m['content'] or ''}" for m in S.chat if not m.get("compare"))
        d1.download_button("Download transcript", transcript, "transcript.md")
        if d2.button("Clear chat"):
            S.chat = []; st.rerun()

# ----------------------------------------------------------------------------- Tab 2
with t2:
    if not S.index or not S.index.kg.g.number_of_nodes():
        st.info("Ingest documents to build the knowledge graph.")
    else:
        kg = S.index.kg
        nodes = sorted(kg.g.nodes)
        bar = st.columns([1, 1, 1, 3])
        if bar[0].button("Enable all"):
            kg.disabled_nodes, kg.disabled_edges = set(), set(); st.rerun()
        if bar[1].button("Disable all"):
            kg.disabled_nodes = set(nodes); st.rerun()
        focus = bar[3].selectbox("Focus entity (highlights its k-hop neighbourhood)", ["(none)"] + nodes)
        with st.expander("Filter entities & relations (changes Graph RAG retrieval)", expanded=False):
            keep = st.multiselect("Active entities (unselect to exclude from retrieval)", nodes,
                                  default=[n for n in nodes if n not in kg.disabled_nodes])
            kg.disabled_nodes = set(nodes) - set(keep)
            edges = sorted({(a, b, d["relation"]) for a, b, d in kg.g.edges(data=True)})
            elabel = lambda e: f"{e[0]} -[{e[2]}]-> {e[1]}"  # noqa: E731
            ekeep = st.multiselect("Active relations", [elabel(e) for e in edges],
                                   default=[elabel(e) for e in edges if e not in kg.disabled_edges])
            kg.disabled_edges = {e for e in edges if elabel(e) not in ekeep}
        rc, v1, v2 = st.columns([2, 1.5, 1.5])
        mode = rc.radio("Rendering", ["Sphere (works in any browser)", "3D WebGL (needs GPU/WebGL)"],
                        help="If the 3D WebGL view is blank or says WebGL is unsupported, use the compatible sphere view.")
        fcs = None if focus == "(none)" else focus
        if mode.startswith("Sphere"):
            az = v1.slider("Rotate (azimuth)", 0, 360, 30)
            el = v2.slider("Tilt (elevation)", -90, 90, 20)
            st.plotly_chart(kg_figure_svg(kg, fcs, S.settings.hp.graph_hops, az, el), width="stretch")
        else:
            st.plotly_chart(kg_figure(kg, fcs, S.settings.hp.graph_hops), width="stretch")
        st.caption(f"{len(nodes) - len(kg.disabled_nodes)} active entities | {len(edges)} relations | layout: Node2Vec-style "
                   "structural embeddings projected onto a sphere. Hover edge midpoints for relation labels; "
                   "grey = excluded from Graph RAG retrieval.")
        if focus != "(none)":
            st.markdown(f"**{focus}** - relations")
            st.dataframe([{"Head": a, "Relation": r, "Tail": b} for a, b, r in edges if focus in (a, b)], hide_index=True)
            st.caption(f"Found in {len(kg.entity_chunks.get(focus, []))} chunk(s).")
        with st.expander("Entity resolution map (aliases -> canonical)"):
            from pwc_rag.entities import ALIASES
            st.dataframe([{"Canonical": c, "Aliases": ", ".join(a for a in al if a != c)} for c, al in ALIASES.items()],
                         hide_index=True)

# ----------------------------------------------------------------------------- Tab 3
with t3:
    st.caption("Toggle nodes to test pipeline variations (applies to the Graph RAG engine). Synthesis is always on.")
    cols = st.columns(len(NODES))
    en = set()
    for c, nme in zip(cols, NODES):
        if c.checkbox(nme, value=nme in S.enabled_nodes, disabled=nme == "synthesis", key=f"n_{nme}"):
            en.add(nme)
    S.enabled_nodes = en | {"synthesis"}
    S.cycle = st.checkbox("Cycle edge: rerank -> hybrid_search (widen hop radius)", value=S.cycle)
    executed = set()
    if S.last and S.last.engine == "Graph RAG":
        executed = {n for n in NODES if any(t.startswith(n) for t in S.last.trace)}
    st.plotly_chart(dag_figure(S.enabled_nodes, S.cycle, executed), width="stretch")
    st.caption("Orange = enabled, grey = disabled, dark ring = executed in the last Graph RAG run. "
               "Disabling hybrid_search falls back to vector-only retrieval.")
    if S.last and S.last.engine == "Graph RAG":
        st.markdown("**Last run trace**")
        st.code("\n".join(S.last.trace))

# ----------------------------------------------------------------------------- Tab 4
with t4:
    hp = S.settings.hp
    st.subheader("Hyperparameters")
    a, b = st.columns(2)
    hp.chunk_size = a.number_input("Chunk size (tokens)", 100, 2000, hp.chunk_size, 50, help="Spec: 600")
    hp.chunk_overlap = a.number_input("Chunk overlap (tokens)", 0, 500, hp.chunk_overlap, 10, help="Spec: 60")
    hp.hyde_n = b.slider("Multi-HyDE passages N", 1, 5, hp.hyde_n, help="Spec: 3")
    hp.graph_hops = b.slider("Graph hop radius k", 0, 4, hp.graph_hops, help="Spec: 2")
    hp.rrf_k = b.number_input("RRF constant", 1, 200, hp.rrf_k, help="Spec: 60")
    hp.top_k = b.slider("Final contexts (top-k)", 1, 10, hp.top_k)
    r1, r2 = st.columns(2)
    if r1.button("Apply & re-index", help="Required after changing chunk size/overlap") and S.raw_docs:
        try:
            hp.validate()
            with st.spinner("Re-indexing..."):
                S.index = build_index(S.raw_docs)
            st.success(f"Re-indexed: {len(S.index.chunks)} chunks")
        except ValueError as e:
            st.error(str(e))
    if r2.button("Restore spec defaults (600/60, N=3, k=2, RRF 60)"):
        S.settings.hp.chunk_size, S.settings.hp.chunk_overlap, S.settings.hp.hyde_n = 600, 60, 3
        S.settings.hp.graph_hops, S.settings.hp.rrf_k = 2, 60
        st.rerun()
    st.caption("hyde_n, hop radius, RRF constant and top-k apply immediately to the next query; chunk settings need re-index.")

    st.subheader("KPI telemetry")
    r = S.last
    _, gold = load_eval("data")
    rcr = None
    if r:
        m = gold_match(r.question)
        rcr = retrieval_context_recall(r, m["facts"]) if m else (S.bench[r.engine]["rcr"] if S.bench and r.engine in S.bench else None)
    k = st.columns(4)
    k[0].metric("Retrieval Context Recall", "n/a" if rcr is None else f"{rcr:.2f}",
                help="Live only if the last question is in data/eval_set.json AND its source documents are loaded; "
                     "otherwise the last Tab 5 benchmark mean (or n/a).")
    k[1].metric("Faithfulness", "n/a" if not r else f"{faithfulness(r):.2f}")
    k[2].metric("Entity-Resolution Acc.", "n/a" if not S.index else f"{entity_resolution_accuracy(S.index, gold):.2f}")
    k[3].metric("End-to-end latency (ms)", "n/a" if not r else f"{r.latency_ms:.0f}")
    if r:
        parts = {kk: vv for kk, vv in r.timings_ms.items() if kk != "total"}
        fig = go.Figure(go.Bar(x=list(parts.values()), y=list(parts), orientation="h", marker_color=ORANGE))
        fig.update_layout(height=60 + 40 * len(parts), margin=dict(l=0, r=0, t=24, b=0), title=f"Latency breakdown - {r.engine} (ms)")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Ask a question in Tab 1 to populate live KPIs.")
    if S.history:
        st.markdown("**Query history (this session)**")
        st.dataframe(S.history[::-1], hide_index=True, width="stretch")
        st.download_button("Download history (JSON)", json.dumps(S.history, indent=2), "query_history.json")

# ----------------------------------------------------------------------------- Tab 5
with t5:
    if not S.index:
        st.info("Ingest documents or load the sample corpus (Tab 1), then run the benchmark.")
    chosen = st.multiselect("Engines to compare", list(ENGINES), default=list(ENGINES))
    st.caption("Runs the bundled gold question set (`data/eval_set.json`) - meaningful on the sample corpus.")
    if st.button("Run benchmark on eval set", type="primary") and S.index and chosen:
        ev, gold = load_eval("data")
        bar = st.progress(0.0, text="Running benchmark...")
        S.bench = evaluate(S.index, ev, gold, engines={k: ENGINES[k] for k in chosen},
                           engine_kwargs={"Graph RAG": {"enabled_nodes": S.enabled_nodes, "enable_cycle": S.cycle}})
        bar.empty()
    if S.bench:
        rows = [{"Engine": k, "RCR": round(v["rcr"], 2), "Faithfulness": round(v["faithfulness"], 2),
                 "Entity-Res. Acc.": v["entity_resolution_accuracy"], "Mean latency (ms)": round(v["latency_ms_mean"], 1),
                 "P95 latency (ms)": round(v["latency_ms_p95"], 1)} for k, v in S.bench.items()]
        st.dataframe(rows, hide_index=True, width="stretch")
        types = sorted({t for v in S.bench.values() for t in v["rcr_by_query_type"]})
        g1, g2 = st.columns(2)
        f1 = go.Figure([go.Bar(name=k, x=types, y=[v["rcr_by_query_type"].get(t, 0) for t in types]) for k, v in S.bench.items()])
        f1.update_layout(barmode="group", title="Retrieval Context Recall by query type", yaxis_range=[0, 1.05], height=340)
        g1.plotly_chart(f1, width="stretch")
        f2 = go.Figure([go.Bar(name="Mean latency (ms)", x=list(S.bench), y=[v["latency_ms_mean"] for v in S.bench.values()],
                               marker_color=ORANGE)])
        f2.update_layout(title="Mean end-to-end latency (ms)", height=340)
        g2.plotly_chart(f2, width="stretch")
        st.subheader("Qualitative matrix (design expectation from the specification)")
        st.table([
            {"Metric": "Simple keyword queries", "Standard": "High precision / low latency", "HyDE": "Moderate (HyDE overhead)", "Graph RAG": "Moderate (graph overhead)"},
            {"Metric": "Multi-hop relational", "Standard": "Poor", "HyDE": "Moderate", "Graph RAG": "Superior (k-hop edges)"},
            {"Metric": "Vocabulary mismatch", "Standard": "Poor", "HyDE": "Superior", "Graph RAG": "High (entity resolution)"},
            {"Metric": "Hallucination risk", "Standard": "Moderate", "HyDE": "Moderate (passage drift)", "Graph RAG": "Lowest (entity grounding)"},
            {"Metric": "Best enterprise use", "Standard": "Single-document lookup (SOPs)", "HyDE": "Vague regulatory inquiries", "Graph RAG": "Complex Tax / M&A due diligence"}])
        st.caption("Measured values come from the bundled eval set on your current configuration - illustrative only; "
                   "re-run with your own corpus and gold questions. Spec latencies (~450/1,200/850 ms) are design targets for a hosted LLM.")
        with st.expander("Per-question results"):
            st.dataframe([{"Engine": k, **{c: (round(x, 2) if isinstance(x, float) else x) for c, x in r.items() if c != "timings_ms"}}
                          for k, v in S.bench.items() for r in v["rows"]], hide_index=True, width="stretch")
        st.download_button("Download report (JSON)", json.dumps(S.bench, indent=2, default=float), "benchmark_report.json")

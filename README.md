# Standard RAG vs HyDE RAG vs Graph RAG - PwC India Advisory (LangChain + LangGraph)

Domain-grounded QA across Tax, Risk Advisory (GRC) and Deal Advisory, with a side-by-side benchmark of three retrieval architectures.

## Architecture
| Spec step | Module |
|---|---|
| 1 Ingestion (PDF/DOCX/JSON/CSV/TXT, provenance metadata) | `pwc_rag/ingestion.py` (UnstructuredPDFLoader, falls back to pypdf) |
| 2 Chunking (600 tok / 60 overlap, clause-aware separators) + entity-relation extraction | `chunking.py`, `entities.py` |
| 3 Sub-word tokenisation (bge tokenizer) + entity normalisation (CBDT = Income Tax Dept = Tax Authority) | `chunking.py`, `entities.py` |
| 4 Embeddings: bge-large-en-v1.5 (1024-d) + Node2Vec-style structural node embeddings | `embeddings.py`, `stores.py` |
| 5 Dual index: vector (numpy exact / Chroma HNSW) + BM25 + graph adjacency | `stores.py`, `index.py` |
| 6 Hybrid engine: LangGraph `query_expansion -> hybrid_search -> rrf_fusion -> rerank -> synthesis`, with a bounded cycle that widens the hop radius when context is thin | `pipelines/graph_rag.py` |

- **Standard RAG**: dense top-k. **HyDE RAG**: N=3 hypothetical passages from distinct regulatory angles, fused with RRF. **Graph RAG**: BM25 + vector + k=2 hop graph lists fused by RRF (constant 60), reranked.
- **CO-STAR prompt + practice-area guardrails** in `prompts.py`; retrieved text is fenced as data, answers must cite `[Sn]`, and gaps are disclosed. Practice-area scoping never returns chunks tagged for another area.
- **KPIs** (`evaluation.py`): Retrieval Context Recall, Faithfulness, Entity-Resolution Accuracy, latency breakdown.
- **UI** (`app.py`, Streamlit): 5 tabs - ingestion & chat, 3D spherical knowledge graph with node/edge toggles, LangGraph workflow toggles, hyperparameters & KPIs, comparative benchmark.

## Run
```bash
pip install -r requirements.txt            # add: pip install sentence-transformers transformers  (for real bge embeddings)
export ANTHROPIC_API_KEY=...               # optional; otherwise an offline extractive LLM is used
python -m pytest -q
python run_benchmark.py                    # CLI benchmark on data/samples
streamlit run app.py
```
Env: `LLM_BACKEND=auto|anthropic|offline`, `EMBEDDING_BACKEND=auto|bge|hash`, `VECTOR_BACKEND=numpy|chroma`, `LLM_MODEL`, `EMBEDDING_MODEL`.

## Honest limitations
- `data/samples` is **synthetic** illustrative text, not real statute. The latency/quality table in the problem statement (450/1200/850 ms) is a design target; this repo *measures* its own numbers, which on the tiny sample corpus are illustrative only.
- Without `sentence-transformers`, a hashing embedding fallback is used (tests/air-gapped demos). Without an LLM key, HyDE passages are template echoes, so HyDE looks weak offline - evaluate with a real LLM.
- Faithfulness is a lexical-overlap proxy; use an LLM judge (e.g. RAGAS) for audit-grade scoring. Neo4j/Qdrant are not wired in: NetworkX and numpy/Chroma are used behind small interfaces.
- Not verified live: the Anthropic call path (no network in the build sandbox). If the LLM fails, answers degrade visibly to extractive mode.
- Uploaded documents may contain client-confidential data/PII; run on approved infrastructure and note text is sent to the configured LLM.

## UI features (app.py)
- Branded theme (`.streamlit/config.toml`), sidebar with live system status (LLM, embeddings, vector store, warnings when running in offline/fallback mode), workspace counters, vector-store switch (numpy/Chroma) and reset.
- Tab 1: drag-and-drop upload with per-file error reporting (corrupt/empty files are reported, not fatal), incremental indexing, document table, example questions per practice area, **side-by-side comparison of all three engines**, highlighted source cards, transcript download.
- Tab 2: Node2Vec-positioned 3D sphere, enable/disable all, per-entity/relation filters that change Graph RAG retrieval, focus-entity k-hop highlight, relation table, alias->canonical resolution map.
- Tab 3: workflow DAG with cycle arc, node + cycle toggles, executed nodes ringed, last-run trace.
- Tab 4: the 4 hyperparameters (restore-spec-defaults button), 4 live KPIs, latency breakdown chart, session query history (JSON download).
- Tab 5: engine multi-select, results table, recall-by-query-type and latency charts, spec qualitative matrix, per-question details, JSON report download.

## Verification
`python -m pytest tests -q` (21) · `python -m pytest e2e -q` (13 Playwright browser tests; needs `pip install playwright && playwright install chromium`; screenshots in `e2e/screenshots/`) · `PYTHONPATH=. python scripts/check_cases.py`. Manual scenarios: `TESTING.md`.

## Hyperparameter tuning (scripts/tune.py)
Sweep on a mixed 16-chunk corpus (samples + test_inputs, unscoped = harder), retrieval recall only, offline LLM:
- Spec defaults (600/60, N=3, k=2, RRF 60, top-k 5) sit on the plateau: Standard and Graph RAG reach 1.00 recall.
- top-k 3 is the smallest value where Graph RAG keeps 1.00 while Standard drops to 0.93.
- RRF constant (10-100) and hop radius (0-3) made no measurable difference on a corpus this small, so they stay at spec values.
- With bge-large-en-v1.5, HyDE recall rose 0.71 -> 0.93 versus the hashing fallback.
Re-run `PYTHONPATH=. python scripts/tune.py` on your real corpus and gold questions before changing defaults.

## LLM providers (OpenRouter / Anthropic / offline)
- **Recommended:** `export OPENROUTER_API_KEY=...` (or copy `.env.example`) and start the app; `LLM_BACKEND=auto` picks it up.
- **From the UI:** sidebar -> *LLM provider* -> OpenRouter -> paste key + model slug -> *Apply & test connection*. The key field is masked,
  held only in server memory for the session, never written to disk or logs, and redacted from any error text. A failed test leaves the active LLM unchanged.
- Use the UI field only on your own machine; on a shared/remote deployment prefer the environment variable (the key travels from your browser to the server).

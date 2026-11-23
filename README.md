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

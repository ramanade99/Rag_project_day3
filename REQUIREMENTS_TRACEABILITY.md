# Requirements traceability (spec -> implementation -> verification)

Status: **Met** = implemented and verified; **Partial** = implemented with a stated gap; **Not met** = not implemented.

| Spec item | Status | Where / how verified | Gap |
|---|---|---|---|
| CO-STAR prompt shared by all 3 pipelines | Met | `prompts.py`; used by `Pipeline.generate`; unit test | - |
| 3 practice areas + guardrails switch | Met | `prompts.PRACTICE_AREAS`, UI dropdown; test + E2E (area isolation) | - |
| Step 1 ingestion PDF/DOCX/JSON/tabular, provenance | Met | `ingestion.py`; E2E uploads PDF, DOCX, TXT | Uses `UnstructuredPDFLoader` if installed, else pypdf |
| Step 2 600/60 recursive chunking, clause-aware; LLM-guided entity-relation extraction | Met | `chunking.py`, `entities.py`; unit tests | LLM-guided extraction only runs with a reachable LLM (rule-based otherwise); not verified live |
| Step 3 HF sub-word tokenisation (bge) + entity normalisation | Partial | `token_counter` uses bge tokenizer if `transformers` installed (else tiktoken); alias map | bge tokenizer not exercised in this sandbox |
| Step 4 bge-large-en-v1.5 1024-d + Node2Vec embeddings | Partial | bge via sentence-transformers when installed; Node2Vec-style (random walk + SVD) embeddings drive the Tab 2 layout | bge model not downloaded/tested here (hash fallback used); node embeddings are not used in retrieval scoring |
| Step 5 dual index (HNSW vector + graph adjacency) | Met | Chroma HNSW (`VECTOR_BACKEND=chroma`, tested via benchmark) / exact numpy; NetworkX adjacency | numpy exact search is the default |
| Step 6 Qdrant/Chroma + Neo4j/NetworkX, LangGraph cyclic | Partial | Chroma + NetworkX; LangGraph with bounded cycle | Qdrant and Neo4j not implemented |
| Tab 1 upload, area dropdown, engine selector, cited chat | Met | E2E (3 engines, `[Sn]` citations) | Upload types PDF/DOCX plus JSON/CSV/TXT/MD |
| Tab 2 3D spherical KG; select/unselect nodes & edges to filter retrieval | Partial | E2E renders WebGL sphere (screenshot); selection via multiselect filters retrieval (`kg.disabled_*`) | Selection is by list, not by clicking in the 3D canvas |
| Tab 3 LangGraph workflow canvas, toggle nodes/edges | Partial | Node toggles + cycle-edge toggle; E2E | Static diagram; linear edges not individually toggleable |
| Tab 4 4 hyperparameters + 4 KPIs | Met | E2E asserts all 4 KPI labels | RCR is live only for questions in the gold set |
| Tab 5 side-by-side comparison | Met | E2E runs benchmark; measured table + per-type chart + spec's qualitative matrix | - |
| Hyperparameters: 600/60, N=3, k=2, RRF 60 | Met | `config.Hyperparameters` defaults, editable in Tab 4 | - |
| KPIs: RCR, faithfulness, ER accuracy, latency breakdown | Met | `evaluation.py` | Faithfulness is a lexical proxy, not an LLM judge |
| Spec latency numbers (~450/1200/850 ms) | Not asserted | Design targets for a real LLM; this repo measures its own | Offline numbers are ms-scale and not comparable |

Verification suites: `python -m pytest tests -q` (21 unit/integration), `python -m pytest e2e -q` (8 Playwright browser tests, screenshots in `e2e/screenshots/`), `PYTHONPATH=. python scripts/check_cases.py` (10 scenarios x 3 engines).

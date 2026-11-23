import json
from pathlib import Path

import pytest

from pwc_rag.chunking import chunk_documents
from pwc_rag.config import Hyperparameters, Settings
from pwc_rag.entities import extract_triples, normalize_entity
from pwc_rag.evaluation import evaluate, faithfulness, load_eval
from pwc_rag.fusion import rrf
from pwc_rag.index import RAGIndex
from pwc_rag.ingestion import load_directory, load_file
from pwc_rag.llm import OfflineLLM
from pwc_rag.pipelines import GraphRAG, HyDERAG, StandardRAG
from pwc_rag.pipelines.base import RAGResult
from pwc_rag.prompts import PRACTICE_AREAS, system_prompt

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def index():
    s = Settings(embedding_backend="hash", llm_backend="offline")
    s.hp.top_k = 2
    idx = RAGIndex.create(s)
    idx.ingest(load_directory(ROOT / "data/samples"))
    return idx


def test_hyperparameter_validation():
    with pytest.raises(ValueError):
        Hyperparameters(chunk_size=100, chunk_overlap=100).validate()
    assert Hyperparameters().validate().rrf_k == 60


def test_rrf_math():
    out = dict(rrf([["a", "b"], ["b", "c"]], k=60))
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert out["a"] == pytest.approx(1 / 61)


def test_entity_normalisation_and_extraction():
    assert normalize_entity("Income Tax Dept") == "CBDT"
    assert normalize_entity("section 194c") == "Section 194C"
    t = extract_triples("The CBIC penalises delayed reporting under the CGST Act.")
    assert any(x.head == "CBIC" and x.relation == "penalises" for x in t)


def test_chunking_respects_size_and_ids():
    from langchain_core.documents import Document
    d = Document(page_content=("Section 1. word " * 400), metadata={"doc_id": "d1"})
    chunks = chunk_documents([d], 100, 10)
    assert len(chunks) > 1 and len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)


def test_ingestion_rejects_unknown_type(tmp_path):
    p = tmp_path / "x.exe"
    p.write_bytes(b"x")
    with pytest.raises(ValueError):
        load_file(p)


def test_json_ingestion(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"slab": "A", "rate": "5%"}]))
    assert "rate: 5%" in load_file(p)[0].page_content


def test_prompt_guardrails():
    for area, cfg in PRACTICE_AREAS.items():
        assert cfg["guardrail"] in system_prompt(area)
    with pytest.raises(ValueError):
        system_prompt("nope")


@pytest.mark.parametrize("cls", [StandardRAG, HyDERAG, GraphRAG])
def test_engines_answer_with_citations(index, cls):
    r = cls(index).run("What is the e-invoicing turnover threshold?", "Tax & Regulatory Services")
    assert r.contexts and "[S" in r.answer and r.latency_ms > 0


def test_graph_rag_multihop_beats_standard(index):
    q = "What is the penalty imposed by the body that issues the e-invoicing notification for delayed IRN reporting?"
    g = GraphRAG(index).run(q, "Tax & Regulatory Services")
    assert any("10,000" in d.page_content for d in g.contexts)


def test_graph_rag_node_toggles(index):
    r = GraphRAG(index, enabled_nodes={"synthesis"}).run("e-invoicing threshold", "Tax & Regulatory Services")
    assert r.answer  # still produces output with everything else disabled
    r = GraphRAG(index, enabled_nodes={"hybrid_search", "rerank"}).run("e-invoicing threshold", "Tax & Regulatory Services")
    assert r.contexts


def test_practice_area_scoping():
    s = Settings(embedding_backend="hash", llm_backend="offline")
    idx = RAGIndex.create(s)
    idx.ingest(load_directory(ROOT / "data/samples/", "Deal Advisory & Corporate Finance"))
    r = StandardRAG(idx).run("indemnity cap", "Tax & Regulatory Services")
    assert r.contexts == []  # nothing tagged for that area -> refuses rather than leaking


def test_faithfulness_flags_uncited_claims(index):
    r = RAGResult("x", "q", "## Answer\n- The penalty is Rs. 99 crore for everyone.\n", index.vectors.docs[:1])
    assert faithfulness(r) == 0.0


def test_node_embeddings_and_hops(index):
    emb = index.kg.node_embeddings(dim=8)
    assert set(emb) == set(index.kg.g.nodes)
    h0, h1, h2 = (index.kg.k_hop(["CGST Act"], k) for k in (0, 1, 2))
    assert h0 == {"CGST Act": 0} and set(h0) <= set(h1) <= set(h2) and len(h1) > 1
    assert max(h1.values()) <= 1 and max(h2.values()) <= 2


def test_benchmark_graph_rag_best_recall(index):
    ev, gold = load_eval(ROOT / "data")
    rep = evaluate(index, ev, gold)
    assert rep["Graph RAG"]["rcr"] >= rep["Standard RAG"]["rcr"]
    assert rep["Graph RAG"]["entity_resolution_accuracy"] == 1.0


def test_offline_llm_never_invents():
    out = OfflineLLM().invoke("s", "QUESTION: unrelated zebra?\n<context>\n[S1] (a)\nTax is due on income.\n</context>")
    assert "Gaps" in out


def test_llm_failure_degrades_visibly(index):
    class Boom:
        name = "boom"
        def invoke(self, *a, **k):
            raise ConnectionError("down")
    p = StandardRAG(index)
    orig, index.llm = index.llm, Boom()
    try:
        r = p.run("e-invoicing turnover threshold", "Tax & Regulatory Services")
    finally:
        index.llm = orig
    assert "LLM unavailable" in r.answer


def test_offline_answer_keeps_rupee_amounts():
    out = OfflineLLM().invoke("s", "QUESTION: What is the e-invoicing turnover threshold?\n<context>\n[S1] (a)\n"
                              "The CBIC mandates e-invoicing where turnover exceeds Rs. 20 crore in any year.\n</context>")
    assert "Rs. 20 crore" in out


def test_hard_wrapped_pdf_lines_do_not_split_amounts():
    from pwc_rag.textutil import split_sentences
    t = "Section 1. The CBIC mandates e-invoicing above Rs.\n20 crore turnover.\nSection 2. Penalty applies."
    assert split_sentences(t)[0].startswith("Section 1.") and any("Rs. 20 crore" in s for s in split_sentences(t))


def test_headings_are_not_entities():
    from pwc_rag.entities import find_entities
    names = [n for n, *_ in find_entities("Section 1. Penalty\nThe CBIC levies a fine under Section 73 of the CGST Act.")]
    assert "Section 1" not in names and "Section 73" in names and "CBIC" in names

"""Hyperparameter sweep on a mixed, harder corpus (data/samples + test_inputs). Retrieval-quality only (RCR, contexts).
Usage: PYTHONPATH=. python scripts/tune.py [--scoped]
Questions whose answer differs between the two synthetic corpora (e-invoicing threshold 5 vs 20 crore) are excluded."""
import argparse, itertools, json, os, statistics
from pathlib import Path
from pwc_rag.config import Settings
from pwc_rag.evaluation import retrieval_context_recall
from pwc_rag.index import RAGIndex
from pwc_rag.ingestion import load_file
from pwc_rag.pipelines import ENGINES

R = Path(__file__).resolve().parent.parent
TAX, GRC, DEAL = "Tax & Regulatory Services", "Risk Advisory & Governance (GRC)", "Deal Advisory & Corporate Finance"
AREA = {"samples/tax_gst_circular.txt": TAX, "samples/direct_tax_cbdt.txt": TAX, "samples/grc_rbi_kyc.txt": GRC,
        "samples/deal_spa.txt": DEAL}
for f in (R / "test_inputs").glob("S*"):
    AREA["../test_inputs/" + f.name] = {"S1": TAX, "S3": TAX, "S4": TAX, "S2": GRC, "S6": GRC, "S5": DEAL}[f.name[:2]]


def questions():
    qs = [q for q in json.loads((R / "data/eval_set.json").read_text()) if q["id"] != "q1"]
    for c in json.loads((R / "test_inputs/cases.json").read_text()):
        if c["expect"] and c["id"] != "C1":
            qs.append({"id": c["id"], "type": "case", "area": c["area"], "question": c["question"], "facts": [c["expect"]]})
    return qs


def build(hp_over, scoped):
    s = Settings()
    for k, v in hp_over.items(): setattr(s.hp, k, v)
    idx = RAGIndex.create(s)
    for rel, area in AREA.items():
        idx.ingest(load_file(R / "data" / rel, area if scoped else None))
    return idx


def score(idx, qs):
    out = {}
    for name, cls in ENGINES.items():
        rc = [retrieval_context_recall(cls(idx).run(q["question"], q["area"]), q["facts"]) for q in qs]
        out[name] = statistics.mean(rc)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scoped", action="store_true"); a = ap.parse_args()
    qs = questions()
    idx0 = build({}, a.scoped)
    print(f"emb={type(idx0.vectors.emb).__name__} llm={idx0.llm.name} chunks={len(idx0.chunks)} questions={len(qs)} scoped={a.scoped}")
    grid = {"top_k": [2, 3, 5, 8], "rrf_k": [10, 30, 60, 100], "graph_hops": [0, 1, 2, 3], "chunk": [(600, 60), (200, 20), (100, 10)]}
    base = {"top_k": 5, "rrf_k": 60, "graph_hops": 2, "chunk_size": 600, "chunk_overlap": 60}
    print("baseline", {k: round(v, 2) for k, v in score(idx0, qs).items()})
    for key, vals in grid.items():
        for v in vals:
            over = dict(base)
            if key == "chunk": over["chunk_size"], over["chunk_overlap"] = v
            else: over[key] = v
            idx = build(over, a.scoped) if key == "chunk" else idx0
            if key != "chunk":
                for k_, v_ in over.items(): setattr(idx.settings.hp, k_, v_)
            sc = score(idx, qs)
            print(f"{key}={v!s:<10}", {k: round(x, 2) for k, x in sc.items()}, f"chunks={len(idx.chunks)}")

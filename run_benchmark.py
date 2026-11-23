"""CLI: build index from data/samples and print the Standard vs HyDE vs Graph RAG comparison."""
from __future__ import annotations

import argparse
import json

from pwc_rag.config import Settings
from pwc_rag.evaluation import evaluate, load_eval
from pwc_rag.index import RAGIndex
from pwc_rag.ingestion import load_directory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--top-k", type=int, default=2, help="small default: the bundled sample corpus has only a few chunks")
    ap.add_argument("--chunk-size", type=int, default=600)
    ap.add_argument("--chunk-overlap", type=int, default=60)
    a = ap.parse_args()
    st = Settings()
    st.hp.top_k, st.hp.chunk_size, st.hp.chunk_overlap = a.top_k, a.chunk_size, a.chunk_overlap
    idx = RAGIndex.create(st)
    n = idx.ingest(load_directory(f"{a.data}/samples"))
    ev, gold = load_eval(a.data)
    rep = evaluate(idx, ev, gold)
    if a.json:
        print(json.dumps(rep, indent=2, default=float))
        return
    print(f"LLM={idx.llm.name}  embeddings={type(idx.vectors.emb).__name__}  chunks={n}")
    print(f"{'Engine':<14}{'RCR':>7}{'Faithful':>10}{'ER-Acc':>8}{'Lat ms':>9}")
    for k, v in rep.items():
        er = "-" if v["entity_resolution_accuracy"] is None else f"{v['entity_resolution_accuracy']:.2f}"
        print(f"{k:<14}{v['rcr']:>7.2f}{v['faithfulness']:>10.2f}{er:>8}{v['latency_ms_mean']:>9.1f}")
        print("   RCR by type:", {t: round(x, 2) for t, x in v["rcr_by_query_type"].items()})


if __name__ == "__main__":
    main()

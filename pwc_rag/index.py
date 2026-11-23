"""Builds the dual index (vector+BM25 and knowledge graph) from raw documents."""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document

from .chunking import chunk_documents
from .config import Settings
from .embeddings import get_embeddings
from .entities import extract_triples
from .llm import LLM, get_llm
from .stores import BM25Index, KnowledgeGraph, VectorIndex


@dataclass
class RAGIndex:
    settings: Settings
    llm: LLM
    vectors: VectorIndex
    bm25: BM25Index = field(default_factory=BM25Index)
    kg: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    chunks: dict[str, Document] = field(default_factory=dict)
    positions: dict[str, int] = field(default_factory=dict)

    @classmethod
    def create(cls, settings: Settings | None = None, llm: LLM | None = None) -> "RAGIndex":
        s = settings or Settings()
        emb = get_embeddings(s.embedding_backend, s.embedding_model)
        return cls(s, llm or get_llm(s.llm_backend, s.llm_model), VectorIndex(emb, s.vector_backend))

    def ingest(self, docs: list[Document]) -> int:
        hp = self.settings.hp.validate()
        chunks = chunk_documents(docs, hp.chunk_size, hp.chunk_overlap, self.settings.embedding_model)
        chunks = [c for c in chunks if c.metadata["chunk_id"] not in self.chunks]  # idempotent re-ingest
        for c in chunks:
            self.positions[c.metadata["chunk_id"]] = len(self.positions)
            self.chunks[c.metadata["chunk_id"]] = c
            cid = c.metadata["chunk_id"]
            self.kg.add_mentions(cid, c.page_content)
            self.kg.add_triples(extract_triples(c.page_content, cid, self.llm))
        self.vectors.add(chunks)
        self.bm25.add(chunks)
        return len(chunks)

    def by_area(self, doc: Document, area: str | None) -> bool:
        return not area or not doc.metadata.get("practice_area") or doc.metadata["practice_area"] == area

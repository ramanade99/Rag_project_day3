"""STEP 4-6 - Dual index: vector index, BM25 sparse index, and entity-relation graph (NetworkX)."""
from __future__ import annotations

import re
from collections import defaultdict

import networkx as nx
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from .entities import Triple, find_entities, normalize_entity

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOK.findall(text.lower())


class VectorIndex:
    """Dense index. 'numpy' = exact cosine (normalised vectors); 'chroma' = Chroma HNSW (cosine)."""

    def __init__(self, embeddings: Embeddings, backend: str = "numpy", collection: str = "pwc_rag"):
        self.emb, self.backend = embeddings, backend
        self.docs: list[Document] = []
        self._mat = np.zeros((0, 0), dtype=np.float32)
        self._chroma = None
        if backend == "chroma":
            import chromadb
            client = chromadb.EphemeralClient()
            try:
                client.delete_collection(collection)
            except Exception:
                pass
            self._chroma = client.create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add(self, docs: list[Document]) -> None:
        if not docs:
            return
        vecs = np.asarray(self.emb.embed_documents([d.page_content for d in docs]), dtype=np.float32)
        if self._chroma is not None:
            base = len(self.docs)
            self._chroma.add(ids=[str(base + i) for i in range(len(docs))], embeddings=vecs.tolist(),
                             documents=[d.page_content for d in docs])
        else:
            self._mat = vecs if self._mat.size == 0 else np.vstack([self._mat, vecs])
        self.docs.extend(docs)

    def search(self, query: str | list[float], k: int = 10) -> list[tuple[int, float]]:
        if not self.docs:
            return []
        q = np.asarray(self.emb.embed_query(query) if isinstance(query, str) else query, dtype=np.float32)
        k = min(k, len(self.docs))
        if self._chroma is not None:
            r = self._chroma.query(query_embeddings=[q.tolist()], n_results=k)
            return [(int(i), 1.0 - float(d)) for i, d in zip(r["ids"][0], r["distances"][0])]
        sims = self._mat @ q
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]


class BM25Index:
    def __init__(self):
        self.docs: list[Document] = []
        self._bm25 = None

    def add(self, docs: list[Document]) -> None:
        self.docs.extend(docs)
        self._bm25 = BM25Okapi([tokenize(d.page_content) or ["_"] for d in self.docs])

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        return [(int(i), float(scores[i])) for i in np.argsort(-scores)[:k] if scores[i] > 0]


class KnowledgeGraph:
    """Entity graph with adjacency (Entity -> Relation -> Target), k-hop traversal and Node2Vec-style embeddings."""

    def __init__(self):
        self.g = nx.MultiDiGraph()
        self.entity_chunks: dict[str, set[str]] = defaultdict(set)
        self.disabled_nodes: set[str] = set()   # UI filtering of retrieval contexts
        self.disabled_edges: set[tuple[str, str, str]] = set()

    def add_triples(self, triples: list[Triple]) -> None:
        for t in triples:
            self.g.add_node(t.head)
            self.g.add_node(t.tail)
            if not any(d.get("relation") == t.relation and d.get("chunk_id") == t.chunk_id
                       for d in (self.g.get_edge_data(t.head, t.tail) or {}).values()):
                self.g.add_edge(t.head, t.tail, relation=t.relation, chunk_id=t.chunk_id)
            if t.chunk_id:
                self.entity_chunks[t.head].add(t.chunk_id)
                self.entity_chunks[t.tail].add(t.chunk_id)

    def add_mentions(self, chunk_id: str, text: str) -> None:
        for name, *_ in find_entities(text):
            self.g.add_node(name)
            self.entity_chunks[name].add(chunk_id)

    def _active(self) -> nx.Graph:
        u = nx.Graph()
        for n in self.g.nodes:
            if n not in self.disabled_nodes:
                u.add_node(n)
        for a, b, d in self.g.edges(data=True):
            if a in u and b in u and (a, b, d["relation"]) not in self.disabled_edges:
                u.add_edge(a, b)
        return u

    def seed_entities(self, query: str) -> list[str]:
        seeds = [n for n, *_ in find_entities(query) if n in self.g and n not in self.disabled_nodes]
        if not seeds:  # fuzzy token overlap with node names
            qt = set(tokenize(query))
            seeds = [n for n in self.g.nodes if n not in self.disabled_nodes and len(set(tokenize(n)) & qt) >= 2]
        return list(dict.fromkeys(seeds))

    def k_hop(self, seeds: list[str], k: int) -> dict[str, int]:
        """Return {entity: hop distance} within k hops of any seed (BFS over active subgraph, undirected)."""
        u = self._active()
        dist: dict[str, int] = {}
        for s in seeds:
            if s not in u:
                continue
            for n, d in nx.single_source_shortest_path_length(u, s, cutoff=k).items():
                dist[n] = min(d, dist.get(n, d))
        return dist

    def chunks_for(self, entity_dist: dict[str, int]) -> dict[str, float]:
        """Chunk -> graph score; closer entities weigh more (1/(1+hop))."""
        scores: dict[str, float] = defaultdict(float)
        for ent, d in entity_dist.items():
            for cid in self.entity_chunks.get(ent, ()):
                scores[cid] += 1.0 / (1 + d)
        return dict(scores)

    def node_embeddings(self, dim: int = 32, walks: int = 10, length: int = 20, seed: int = 7) -> dict[str, np.ndarray]:
        """Structural embeddings from random walks + SVD of the co-visit matrix (Node2Vec-style, dependency-free)."""
        u = self.g.to_undirected()
        nodes = list(u.nodes)
        if not nodes:
            return {}
        idx = {n: i for i, n in enumerate(nodes)}
        rng = np.random.default_rng(seed)
        co = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
        for n in nodes:
            for _ in range(walks):
                cur, window = n, [n]
                for _ in range(length):
                    nb = list(u.neighbors(cur))
                    if not nb:
                        break
                    cur = nb[rng.integers(len(nb))]
                    for w in window[-3:]:
                        co[idx[w], idx[cur]] += 1
                        co[idx[cur], idx[w]] += 1
                    window.append(cur)
        ppmi = np.log1p(co)
        U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
        d = min(dim, len(nodes))
        emb = U[:, :d] * np.sqrt(S[:d])
        return {n: emb[idx[n]] for n in nodes}

    def resolution_map(self) -> dict[str, str]:
        return {a: c for a, c in ((a, normalize_entity(a)) for a in self.g.nodes)}

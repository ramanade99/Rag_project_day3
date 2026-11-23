"""STEP 3/4 - Embeddings. bge-large-en-v1.5 (1024-d) when available; deterministic hashing fallback."""
from __future__ import annotations

import hashlib
import re

import numpy as np
from langchain_core.embeddings import Embeddings

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbeddings(Embeddings):
    """Offline, deterministic bag-of-words + bigram hashing embedding (L2-normalised).

    Intended for tests / air-gapped demos only; use BGE for real retrieval quality.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        toks = _TOKEN.findall(text.lower())
        v = np.zeros(self.dim, dtype=np.float32)
        for t in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
            h = int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class BGEEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, text):
        return self.model.encode(BGE_QUERY_PREFIX + text, normalize_embeddings=True).tolist()


def get_embeddings(backend: str = "auto", model_name: str = "BAAI/bge-large-en-v1.5") -> Embeddings:
    if backend in ("auto", "bge"):
        try:
            return BGEEmbeddings(model_name)
        except Exception:
            if backend == "bge":
                raise
    return HashingEmbeddings()

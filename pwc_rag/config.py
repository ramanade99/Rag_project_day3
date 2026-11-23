"""Central configuration: the 4 hyperparameters and runtime settings."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict


@dataclass
class Hyperparameters:
    chunk_size: int = 600          # tokens
    chunk_overlap: int = 60        # tokens
    hyde_n: int = 3                # Multi-HyDE synthetic passages
    graph_hops: int = 2            # k-hop radius
    rrf_k: int = 60                # RRF constant (alpha)
    top_k: int = 5                 # final contexts passed to the LLM

    def validate(self) -> "Hyperparameters":
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        if self.hyde_n < 1 or self.graph_hops < 0 or self.rrf_k < 1 or self.top_k < 1:
            raise ValueError("invalid hyperparameter value")
        return self

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Settings:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    # "auto": use bge if sentence-transformers is installed, else hashing fallback.
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "auto")  # auto|bge|hash
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
    llm_backend: str = os.getenv("LLM_BACKEND", "auto")  # auto|anthropic|offline
    vector_backend: str = os.getenv("VECTOR_BACKEND", "numpy")  # numpy|chroma
    persist_dir: str = os.getenv("PERSIST_DIR", "storage")
    hp: Hyperparameters = field(default_factory=Hyperparameters)

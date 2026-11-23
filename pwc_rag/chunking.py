"""STEP 2/3 - Token-aware semantic chunking that avoids splitting legal clauses.

Uses RecursiveCharacterTextSplitter with a token length function. The HF tokenizer for
bge-large-en-v1.5 is used when available (matches its 512-token limit awareness); otherwise tiktoken.
Separators prefer clause / section boundaries before sentence and word boundaries.
"""
from __future__ import annotations

import functools

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Legal structure first: sections, numbered clauses, paragraphs, then sentences.
CLAUSE_SEPARATORS = [r"\n(?=Section\s+\d)", r"\n(?=Clause\s+\d)", r"\n(?=Rule\s+\d)", r"\n(?=\d+(\.\d+)*[.)]\s)",
                     "\n\n", "\n", r"(?<=[.;])\s", " ", ""]


@functools.lru_cache(maxsize=2)
def token_counter(model_name: str = "BAAI/bge-large-en-v1.5"):
    try:  # sub-word tokenizer matching the embedding model
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        return lambda s: len(tok.encode(s, add_special_tokens=False))
    except Exception:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return lambda s: len(enc.encode(s))
        except Exception:
            return lambda s: max(1, int(len(s.split()) * 1.3))


def chunk_documents(docs: list[Document], chunk_size: int = 600, chunk_overlap: int = 60,
                    model_name: str = "BAAI/bge-large-en-v1.5") -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=CLAUSE_SEPARATORS, is_separator_regex=True, keep_separator=True,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=token_counter(model_name))
    chunks = splitter.split_documents(docs)
    counts: dict[str, int] = {}
    for c in chunks:
        did = c.metadata.get("doc_id", "x")
        counts[did] = counts.get(did, 0) + 1
        c.metadata["chunk_id"] = f"{did}-{counts[did]:04d}"
    return chunks

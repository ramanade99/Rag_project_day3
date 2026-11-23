"""Reciprocal Rank Fusion across sparse / dense / graph rank lists."""
from __future__ import annotations


def rrf(rank_lists: list[list[str]], k: int = 60, weights: list[float] | None = None) -> list[tuple[str, float]]:
    """score(d) = sum_i w_i / (k + rank_i(d)), rank starting at 1."""
    weights = weights or [1.0] * len(rank_lists)
    scores: dict[str, float] = {}
    for w, lst in zip(weights, rank_lists):
        for rank, key in enumerate(dict.fromkeys(lst), 1):
            scores[key] = scores.get(key, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

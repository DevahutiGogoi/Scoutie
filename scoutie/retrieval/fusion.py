"""Combines the three routes into one ranked pool via weighted Reciprocal Rank Fusion.

RRF works purely on each route's rank position, avoiding the scale mismatch of averaging raw
BM25/IDF-overlap/cosine-similarity scores directly.
"""

from __future__ import annotations

from scoutie.config.thresholds import RRF_K


def fuse(route_scores: dict[str, list[tuple[str, float]]], weights: dict[str, float]) -> list[tuple[str, float]]:
    """score(d) = sum over routes r of weight_r / (RRF_K + rank_r(d)), rank_r 1-indexed."""
    scores: dict[str, float] = {}
    for route, scored in route_scores.items():
        weight = weights.get(route, 0.0)
        if weight == 0.0:
            continue
        for rank, (asin, _) in enumerate(scored, start=1):
            scores[asin] = scores.get(asin, 0.0) + weight / (RRF_K + rank)
    fused = list(scores.items())
    fused.sort(key=lambda pair: (-pair[1], pair[0]))
    return fused

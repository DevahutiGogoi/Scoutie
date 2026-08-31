"""Deterministic pass after rank.py: any candidate satisfying every stated hard slot gets
boosted above any candidate that violates one, regardless of incoming rank order, and the
satisfying bucket is itself sub-sorted by descending match-count -- how many currently-stated
slots a candidate genuinely MATCHES, not just doesn't violate -- so a candidate that confirms
every stated slot outranks one that merely doesn't contradict any of them."""

from __future__ import annotations

from scoutie.state import SessionState
from scoutie.understanding.constraint_engine import match_text, stated_hard_slots


def _match_count(stated: list[tuple[str, str]], product: dict, track: str) -> int:
    return sum(1 for attribute, value in stated if match_text(attribute, value, product, track) == "match")


def apply(ranked: list[tuple[str, float]], state: SessionState, products: dict[str, dict]) -> list[tuple[str, float]]:
    stated = stated_hard_slots(state)
    if not stated:
        return ranked

    satisfying: list[tuple[str, float]] = []
    violating: list[tuple[str, float]] = []
    for asin, score in ranked:
        product = products[asin]
        if all(match_text(attribute, value, product, state.track) != "violate" for attribute, value in stated):
            satisfying.append((asin, score))
        else:
            violating.append((asin, score))
    if not satisfying:
        return satisfying + violating

    graded = sorted(
        enumerate(satisfying),
        key=lambda indexed: (-_match_count(stated, products[indexed[1][0]], state.track), indexed[0]),
    )
    return [item for _, item in graded] + violating

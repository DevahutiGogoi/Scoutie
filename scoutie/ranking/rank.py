"""Ranks the incoming candidate pool with a deterministic weighted local scoring function
(no model call anywhere in this pipeline -- see CLAUDE.md's standing no-LLM rule). Four
components, each normalized to [0, 1] before the weighted sum -- naively combining
differently-scaled signals would repeat the mistake RRF fusion avoids for route scores:
  - fused: the incoming RRF-fused retrieval score, min-max normalized within this turn's pool.
  - soft_slot: decay-weighted fraction of the session's currently stated (non-category) slots
    this candidate genuinely MATCHES (not just "doesn't violate"). Each slot's contribution is
    weighted by its current decayed weight (SLOT_DECAY_RATE/SLOT_DECAY_FLOOR,
    config/thresholds.py) based on turns elapsed since it was last (re)stated -- a slot
    reaffirmed or just disclosed this turn counts fully; one unreaffirmed for several turns
    counts less, down to SLOT_DECAY_FLOOR. Decay is confined entirely to this ranking
    tie-breaker; it never touches constraint_engine.py's hard-filter or guarantee_pass.py's
    boost, both driven purely by Slot.status, never Slot.weight.
  - profile: fraction of the session's profile_prior["preference_tags"] that appear in the
    candidate's searchable text.
  - popularity: average_rating, Bayesian-shrunk toward the catalog mean by rating_number (a
    thinly-reviewed product's rating is pulled toward CATALOG_MEAN_RATING; a well-established
    one is barely moved), rating-floor-gated and linearly scaled to [0, 1].

guarantee_pass.py still runs immediately after this, unconditionally, and still boosts every
hard-slot-satisfying candidate above every violator regardless of the order this stage produces --
this stage only affects relative order WITHIN each of guarantee_pass's two buckets, never across
them.
"""

from __future__ import annotations

from scoutie.config.thresholds import (
    CATALOG_MEAN_RATING,
    POPULARITY_BAYESIAN_PRIOR_STRENGTH,
    POPULARITY_RATING_FLOOR,
    RANK_WEIGHTS,
    SLOT_DECAY_FLOOR,
    SLOT_DECAY_RATE,
)
from scoutie.state import SessionState, Slot
from scoutie.understanding.constraint_engine import match_text, searchable_text, stated_hard_slots


def _normalize_fused(candidates: list[tuple[str, float]]) -> dict[str, float]:
    if not candidates:
        return {}
    values = [score for _, score in candidates]
    lo, hi = min(values), max(values)
    if hi <= lo:
        return {asin: 1.0 for asin, _ in candidates}
    return {asin: (score - lo) / (hi - lo) for asin, score in candidates}


def _decayed_weight(slot: Slot, current_turn: int) -> float:
    if slot.last_updated_turn is None:
        return slot.weight
    turns_since_update = current_turn - slot.last_updated_turn
    if turns_since_update <= 0:
        return slot.weight
    decayed = slot.weight * (SLOT_DECAY_RATE ** turns_since_update)
    return max(SLOT_DECAY_FLOOR, decayed)


def _soft_slot_score(state: SessionState, product: dict) -> float:
    stated = stated_hard_slots(state)
    if not stated:
        return 0.0
    total_weight = 0.0
    matched_weight = 0.0
    for attribute, value in stated:
        slot = state.slots[attribute]
        weight = _decayed_weight(slot, state.turn_count)
        total_weight += weight
        if match_text(attribute, value, product, state.track) == "match":
            matched_weight += weight
    return matched_weight / total_weight if total_weight > 0 else 0.0


def _profile_score(state: SessionState, product: dict) -> float:
    tags = state.profile_prior.get("preference_tags") or []
    if not tags:
        return 0.0
    text = searchable_text(product).lower()
    hits = sum(1 for tag in tags if str(tag).lower() in text)
    return hits / len(tags)


def _shrunk_rating(product: dict) -> float | None:
    rating = product.get("average_rating")
    if rating is None:
        return None
    count = product.get("rating_number")
    if not isinstance(count, (int, float)) or count < 0:
        count = 0.0
    return (count * rating + POPULARITY_BAYESIAN_PRIOR_STRENGTH * CATALOG_MEAN_RATING) / (
        count + POPULARITY_BAYESIAN_PRIOR_STRENGTH
    )


def _popularity_score(product: dict) -> float:
    shrunk = _shrunk_rating(product)
    if shrunk is None:
        return 0.0
    if shrunk < POPULARITY_RATING_FLOOR:
        return 0.0
    return max(0.0, min(1.0, (shrunk - POPULARITY_RATING_FLOOR) / (5.0 - POPULARITY_RATING_FLOOR)))


def rank_weighted(
    candidates: list[tuple[str, float]], state: SessionState, products: dict[str, dict]
) -> list[tuple[str, float]]:
    if not candidates:
        return candidates
    fused_norm = _normalize_fused(candidates)
    scored: list[tuple[str, float]] = []
    for asin, _ in candidates:
        product = products[asin]
        score = (
            RANK_WEIGHTS["fused"] * fused_norm.get(asin, 0.0)
            + RANK_WEIGHTS["soft_slot"] * _soft_slot_score(state, product)
            + RANK_WEIGHTS["profile"] * _profile_score(state, product)
            + RANK_WEIGHTS["popularity"] * _popularity_score(product)
        )
        scored.append((asin, score))
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored

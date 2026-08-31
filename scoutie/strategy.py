"""Ask vs. guess, and which attribute to ask, each turn."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import scoutie.config.thresholds as thresholds
from scoutie.config.thresholds import (
    ASKABLE_ATTRIBUTES,
    ASK_TURN_CAP,
    DEFAULT_MESSAGE,
    MIN_POOL_FOR_ASK_HEURISTIC,
    RECOMMENDATIONS_RETURN_COUNT,
)
from scoutie.state import SessionState
from scoutie.understanding.constraint_engine import ATTRIBUTE_BUCKET_FUNCTIONS, DomainSignal


def should_suppress_asking(state: SessionState) -> bool:
    if state.turn_count >= ASK_TURN_CAP:
        return True
    return not any(state.slots[attribute].status == "unknown" for attribute in ASKABLE_ATTRIBUTES)


def _bucket_budget(product: dict) -> str | None:
    price = product.get("price")
    if price in (None, ""):
        return None
    try:
        price = float(price)
    except (TypeError, ValueError):
        # A handful of catalog rows have a non-numeric price string (e.g. "-", "from 12.99").
        # A non-numeric price has no determinable budget bucket, same as a missing one -- None,
        # not a crash.
        return None
    if price < 15:
        return "under_15"
    if price < 30:
        return "15_30"
    if price < 50:
        return "30_50"
    if price < 100:
        return "50_100"
    return "over_100"


def _bucket_feature(product: dict) -> str | None:
    features = product.get("features")
    if isinstance(features, list) and features:
        words = str(features[0]).strip().lower().split()
        return " ".join(words[:3]) if words else None
    return None


# material/color/size/style/use_case buckets live in constraint_engine.py (shared with
# match_text()'s violate check -- one source of truth for "what is this product's X").
BUCKET_FUNCTIONS = {
    **ATTRIBUTE_BUCKET_FUNCTIONS,
    "budget": _bucket_budget,
    "feature": _bucket_feature,
}


def attribute_entropy(attribute: str, domain_asins: list[str], products: dict[str, dict]) -> float:
    """Shannon entropy (bits) of this attribute's value distribution across the candidate pool.
    Products with no determinable value for this attribute are excluded from the distribution
    entirely -- absence isn't a signal, not counted as a value of its own."""
    bucket_fn = BUCKET_FUNCTIONS[attribute]
    counts = Counter(bucket_fn(products[asin]) for asin in domain_asins)
    counts.pop(None, None)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def choose_attribute_to_ask_entropy(
    state: SessionState, domain_asins: list[str], products: dict[str, dict]
) -> str | None:
    """Picks the unknown-status slot with the highest Shannon entropy across the current
    candidate pool (real information gain). Below MIN_POOL_FOR_ASK_HEURISTIC, entropy is too
    noisy to be meaningful, so it falls back to "feature" (the fuzziest, highest-recall bucket)."""
    candidates = [attribute for attribute in ASKABLE_ATTRIBUTES if state.slots[attribute].status == "unknown"]
    if not candidates:
        return None
    if len(domain_asins) < MIN_POOL_FOR_ASK_HEURISTIC:
        return "feature" if "feature" in candidates else candidates[0]

    best_attribute = candidates[0]
    best_entropy = -1.0
    for attribute in candidates:
        entropy = attribute_entropy(attribute, domain_asins, products)
        if entropy > best_entropy:
            best_attribute, best_entropy = attribute, entropy
    return best_attribute


@dataclass
class StrategyDecision:
    ask_attribute: str | None
    message: str
    recommendations: list[tuple[str, float]]


def build_message(ask_attribute: str | None, domain_signal: DomainSignal) -> str:
    if ask_attribute:
        return f"To narrow this down, could you tell me more about {ask_attribute.replace('_', ' ')}?"
    if domain_signal.collision:
        return "I'm having a hard time matching everything you've told me exactly -- here's my best-effort list."
    if domain_signal.over_generality:
        return "There's a lot to choose from here -- here are some options while I narrow things down."
    return DEFAULT_MESSAGE


def decide(
    state: SessionState,
    ranked: list[tuple[str, float]],
    domain_signal: DomainSignal,
    products: dict[str, dict],
) -> StrategyDecision:
    # Always populate recommendations, every turn, whether or not also asking -- no scoring
    # penalty for an early guess that misses, only upside if it happens to land.
    recommendations = ranked[:RECOMMENDATIONS_RETURN_COUNT]
    domain_asins = [asin for asin, _ in ranked]

    ask_attribute = None
    if not should_suppress_asking(state):
        # customer_reply()'s match filter collapses to "any not-yet-disclosed constraint, any
        # type" when the asked attribute is "other" -- asking it up to ASK_OTHER_MAX_COUNT times
        # per session reveals substantially more than asking a specific attribute would. Read via
        # module reference (not a plain `from ... import`) so the value stays live if ever tuned
        # at runtime.
        if state.other_ask_count < thresholds.ASK_OTHER_MAX_COUNT:
            ask_attribute = "other"
            state.other_ask_count += 1
        else:
            ask_attribute = choose_attribute_to_ask_entropy(state, domain_asins, products)

    message = build_message(ask_attribute, domain_signal)
    return StrategyDecision(ask_attribute=ask_attribute, message=message, recommendations=recommendations)

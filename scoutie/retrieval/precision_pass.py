"""Narrows the fused pool to the actually-scored window, and rescues a strong CategoryRoute
candidate that the keyword-dominant fused ranking would otherwise bury outside it."""

from __future__ import annotations

from scoutie.state import SessionState
from scoutie.understanding.constraint_engine import match_text, stated_hard_slots


def truncate(fused: list[tuple[str, float]], top_k: int) -> list[tuple[str, float]]:
    return fused[:top_k]


def category_rescue(
    ranked: list[tuple[str, float]],
    category_picks: list[tuple[str, float]],
    state: SessionState,
    products: dict[str, dict],
    rescue_count: int,
    top_k: int,
) -> list[tuple[str, float]]:
    """Reserves up to `rescue_count` of the final top-`top_k` slots (the ones actually scored) for
    CategoryRoute's own best pick(s) that aren't already there.

    Guardrail: a candidate is only ever rescued if it's already present somewhere in `ranked` (i.e.
    it passed retrieval/fusion/precision_pass on its own merit, just outside the visible top_k) and
    does not violate any currently-stated hard slot -- never invents a brand-new candidate, and
    never lets a rescue break guarantee_pass's non-violation invariant. Displaces the LOWEST-ranked
    incumbent(s) of the current top_k window, not a fixed slot, so a category-found candidate never
    bumps something more confidently placed.
    """
    if rescue_count <= 0 or not category_picks:
        return ranked
    ranked_map = dict(ranked)
    top_ids = {asin for asin, _ in ranked[:top_k]}
    stated = stated_hard_slots(state)

    rescues: list[str] = []
    for asin, _ in category_picks:
        if len(rescues) >= rescue_count:
            break
        if asin in top_ids or asin not in ranked_map:
            continue
        product = products[asin]
        if any(match_text(attribute, value, product, state.track) == "violate" for attribute, value in stated):
            continue
        rescues.append(asin)
    if not rescues:
        return ranked

    rescue_set = set(rescues)
    kept_top = [pair for pair in ranked[:top_k] if pair[0] not in rescue_set][: top_k - len(rescues)]
    rescued_pairs = [(asin, ranked_map[asin]) for asin in rescues]
    rest = [pair for pair in ranked[top_k:] if pair[0] not in rescue_set]
    return kept_top + rescued_pairs + rest

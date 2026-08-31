"""The load-bearing module: hard-filter matching, over-generality/collision signals.

Pass-two coverage fix (2026-08-27 round 2, see PROGRESS.md): the original match_text() only
ever returned "violate" for the budget slot, by design -- see match_text_generic()'s docstring
below for the sparse-metadata guardrail that motivated it. The audit in
results/match_text_audit_2026-08-27.json confirmed this at 0% violate rate across 6,960-43,320
per-attribute evaluations for material/color/size/style/use_case/feature, which meant
guarantee_pass's boost/demote and precision_pass's Buying hard-filter were effectively inert for
8 of 9 attribute types. match_text() is now attribute-aware: for material/color/size/style/
use_case it compares the stated value against the product's own *specific, determinable* value
for that attribute (a structured `details` field, or an explicit regex/keyword hit) -- and still
returns "unknown", never "violate", whenever the product simply doesn't have a determinable value
for that attribute. This preserves the original guardrail exactly (absence is never treated as
contradiction) while finally letting a genuine, present-and-different value be detected. The old
attribute-blind version is kept as match_text_generic() -- not deleted -- as the pass-one
fallback, and is still what `feature`/`brand`/`category`/`other` and any unhandled attribute use.

Round-2 follow-up (2026-08-27): the attribute-aware branch above is gated to `track == "buying"`
via match_text()'s required `track` argument -- it helped Buying cleanly but measurably hurt
Browsing (see match_text()'s docstring and PROGRESS.md), so every other track uses
match_text_generic() instead, exactly as pre-round-2.

Performance note: "domain" here means the CURRENT retrieval candidate pool (a few dozen items,
already narrowed by routes/fusion/precision_pass), not the full 50,000-product catalog. Scanning
the full catalog against match_text() every turn (up to ~2,000 turns across a public-set run)
would be prohibitively slow for a signal that's only ever used heuristically by the orchestrator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scoutie.text_utils import COLOR_RE, MATERIAL_RE, _terms, _text
from scoutie.config.thresholds import (
    BUDGET_DOMAIN_SLACK,
    BUDGET_EXACT_EPSILON,
    CATEGORY_HARD_MATCH_ENABLED,
    COLLISION_DOMAIN_SIZE,
    FUZZY_TOKEN_OVERLAP_THRESHOLD,
    OVER_GENERALITY_DOMAIN_SIZE,
)
from scoutie.state import SessionState

SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
BUDGET_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")
STYLE_KEYWORDS = ("sleeve", "neck", "fit", "crew", "v-neck", "hooded", "sleeveless")
USE_CASE_KEYWORDS = ("hiking", "running", "gym", "winter", "outdoor", "work")


def searchable_text(product: dict) -> str:
    return " ".join(_text(product.get(field)) for field in SEARCH_FIELDS)


def _looks_like_budget(raw_value: str) -> bool:
    lowered = raw_value.lower()
    return "$" in raw_value or "budget" in lowered or re.search(r"(?:<=|under)\s*\d", lowered) is not None


def _needle(raw_value: str) -> str:
    cleaned = raw_value.strip()
    if ":" in cleaned:
        cleaned = cleaned.split(":", 1)[1].strip()
    return cleaned.lower()


# --- Per-attribute "what does this specific product's value look like" extractors. ---
# Reused by _attribute_match() below AND by orchestration/strategy.py's ask-attribute
# distinct-value heuristic (imported from here as ATTRIBUTE_BUCKET_FUNCTIONS): one source of
# truth for "what is this product's material/color/size/style/use_case", not two.


def _bucket_material(product: dict) -> str | None:
    details = product.get("details") or {}
    if isinstance(details, dict) and details.get("Material"):
        return str(details["Material"]).strip().lower()
    match = MATERIAL_RE.search(searchable_text(product))
    return match.group(1).lower() if match else None


def _bucket_color(product: dict) -> str | None:
    details = product.get("details") or {}
    if isinstance(details, dict) and details.get("Color"):
        return str(details["Color"]).strip().lower()
    match = COLOR_RE.search(searchable_text(product))
    return match.group(1).lower() if match else None


def _bucket_size(product: dict) -> str | None:
    # Round 14 Tier 2 item #8: no free-text fallback, unlike material/color/style. Checked before
    # building one: 11.6% of the catalog (5,799/50,000) lacks all three structured keys, but
    # across the real 200-session public set, ZERO Buying-track turns ever have "size" as a
    # stated hard constraint -- classify_constraint()'s size vocabulary just doesn't get
    # triggered by this catalog's actual disclosed-constraint text. The abstract catalog gap
    # exists but is never exercised by anything scored. Left as-is; see PROGRESS.md.
    details = product.get("details") or {}
    if isinstance(details, dict):
        for key in ("Size", "Size Name", "Department"):
            if details.get(key):
                return str(details[key]).strip().lower()
    return None


def _bucket_style(product: dict) -> str | None:
    """Round 14 Tier 2 item #7: a proxy check found the free-text STYLE_KEYWORDS scan below
    disagrees with the product's own structured `details.Style` value 94.7% of the time it fires
    (see PROGRESS.md) -- STYLE_KEYWORDS is generic apparel-fit vocabulary ("sleeve", "neck",
    "fit"...) that hits constantly as an incidental substring across this catalog's much broader
    style vocabulary (jewelry, bags, accessories), almost never the product's actual style.
    Tried removing the fallback (return None, i.e. "unknown", when there's no structured value --
    the same "absence is never treated as contradiction" guardrail this module applies everywhere
    else) and verified against the full evaluator: it REGRESSED the composite
    (recommended_technical_score 0.707652 -> 0.704258, driven by buying's hit_rate dropping and
    mttc worsening in boundary/browsing/buying) despite the strong data-quality argument for it --
    the noisy guess apparently still helps guarantee_pass.py discriminate more than "unknown"
    (which lets more candidates through as trivially "satisfying") does. Reverted; kept as the
    checkpoint-safe version. A smarter fix (e.g. only trusting the scan when a keyword clearly
    dominates, or a different keyword vocabulary matched to this catalog's real style space) is a
    plausible future round, but a blind "just make it not lie" fix measurably isn't the answer."""
    details = product.get("details") or {}
    if isinstance(details, dict) and details.get("Style"):
        return str(details["Style"]).strip().lower()
    text = searchable_text(product).lower()
    for keyword in STYLE_KEYWORDS:
        if keyword in text:
            return keyword
    return None


def _bucket_use_case(product: dict) -> str | None:
    text = searchable_text(product).lower()
    for keyword in USE_CASE_KEYWORDS:
        if keyword in text:
            return keyword
    return None


def _bucket_category(product: dict) -> str | None:
    """The product's own category path, lowercased, as free text -- deliberately not reduced to
    a single token like the other buckets, since _attribute_match()'s substring-then-fuzzy-overlap
    check already handles "is this short stated phrase contained in this longer path" correctly
    without needing a bucket per exact category. None only when the catalog row has no category
    path at all (rare in practice, but the same "absence is never contradiction" guardrail as
    every other attribute here).

    Round 19 fix: the catalog's own top-level department label is dropped before matching. This
    frozen catalog has only two such labels across its ~50,000 rows ("Clothing, Shoes & Jewelry"
    and "Shoe, Jewelry & Watch Accessories" -- confirmed by scanning the file directly), and both
    happen to contain "shoes"/"shoe" as boilerplate. Confirmed live: a shopper who stated
    category "shoes" got a handbag as their final, "satisfies every constraint" pick, because
    EVERY row in the catalog trivially substring-matches "shoes" through this shared department
    label alone, regardless of what the product actually is. Keeping only categories[1:] (the
    genuinely product-specific levels -- "Women", "Handbags & Wallets", "Fashion Backpacks", etc.)
    restores real discriminating power without changing the guardrail: still "unknown", never
    "violate", whenever a row has no more than that one boilerplate level.
    """
    categories = product.get("categories")
    if isinstance(categories, list) and len(categories) > 1:
        categories = categories[1:]
    text = _text(categories).strip().lower()
    return text or None


ATTRIBUTE_BUCKET_FUNCTIONS = {
    "material": _bucket_material,
    "color": _bucket_color,
    "size": _bucket_size,
    "style": _bucket_style,
    "use_case": _bucket_use_case,
}

# Category is handled separately from ATTRIBUTE_BUCKET_FUNCTIONS above (not merged in) because it
# needs its own gate (CATEGORY_HARD_MATCH_ENABLED) and its own consumer (stated_hard_slots()
# below), independent of every other attribute's already-track-gated match_text() behavior --
# category matching isn't gated to track=="buying" the way material/color/etc. are, since a
# browsing-track shopper's stated category is just as real a signal as a buying-track shopper's.
CATEGORY_BUCKET_FUNCTION = _bucket_category


def _bucket_match(bucket_fn, raw_value: str, product: dict) -> str:
    """match | violate | unknown against a single bucket_fn's "what is this product's value for
    this attribute" extraction. Only ever "violate" when the product has a specific, determinable
    value that clearly doesn't overlap the stated value; still "unknown" -- never "violate" --
    whenever the product simply has no determinable value, preserving the original guardrail.
    """
    product_value = bucket_fn(product)
    if product_value is None:
        return "unknown"
    needle = _needle(raw_value)
    if not needle:
        return "unknown"
    if needle in product_value or product_value in needle:
        return "match"
    needle_terms = set(_terms(needle))
    product_terms = set(_terms(product_value))
    if not needle_terms or not product_terms:
        return "unknown"
    overlap = len(needle_terms & product_terms) / len(needle_terms)
    return "match" if overlap >= FUZZY_TOKEN_OVERLAP_THRESHOLD else "violate"


def _match_budget(raw_value: str, product: dict) -> str:
    number_match = BUDGET_NUMBER_RE.search(raw_value)
    price = product.get("price")
    if not number_match or price in (None, ""):
        return "unknown"
    amount = float(number_match.group(1))
    try:
        price = float(price)
    except (TypeError, ValueError):
        # A handful of catalog rows have a non-numeric price string (e.g. "-", "from 12.99").
        # A non-numeric price has no determinable budget value, same as a missing one -- "unknown",
        # not a crash (same guard as strategy._bucket_budget()).
        return "unknown"
    if abs(price - amount) <= BUDGET_EXACT_EPSILON:
        return "match"
    if abs(price - amount) <= amount * BUDGET_DOMAIN_SLACK:
        return "unknown"
    return "violate"


def match_text_generic(raw_value: str | None, product: dict) -> str:
    """Pass-one fallback (kept, not deleted, per CLAUDE.md's Definition of Done #4): the original
    attribute-blind haystack-containment / fuzzy-token-overlap check. Never returns "violate" for
    anything but budget-shaped raw values -- see module docstring for why this was written that
    way, and why match_text() below now goes further for the attributes it's safe to.
    """
    if not raw_value or not raw_value.strip():
        return "unknown"
    if _looks_like_budget(raw_value):
        return _match_budget(raw_value, product)

    haystack = searchable_text(product).lower()
    needle = _needle(raw_value)
    if needle and needle in haystack:
        return "match"

    needle_terms = set(_terms(raw_value))
    if not needle_terms:
        return "unknown"
    haystack_terms = set(_terms(haystack))
    overlap = len(needle_terms & haystack_terms) / len(needle_terms)
    return "match" if overlap >= FUZZY_TOKEN_OVERLAP_THRESHOLD else "unknown"


def match_text(attribute: str, raw_value: str | None, product: dict, track: str) -> str:
    """Returns "match" | "violate" | "unknown". See module docstring for the pass-two coverage
    fix: budget and material/color/size/style/use_case can now genuinely violate; every other
    attribute (feature/brand/category/other) falls back to match_text_generic(), which can't.

    Round-2 follow-up (2026-08-27, see PROGRESS.md "Buying-track gating for match_text()"): the
    attribute-aware violate-detection for material/color/size/style/use_case helped Buying
    cleanly (hit_rate 0.6375 -> 0.75) but hurt Browsing (hit_rate 0.6 -> 0.5875) -- Browsing's
    earlier, noisier disclosures are more exposed to a binary satisfying/violating partition than
    Buying's. Gated to `track == "buying"` only; every other track falls back to
    match_text_generic()'s never-violate-except-budget behavior, exactly as it did before the
    round-2 coverage fix.
    """
    if not raw_value or not raw_value.strip():
        return "unknown"
    if attribute == "budget" or _looks_like_budget(raw_value):
        return _match_budget(raw_value, product)
    # Not track-gated like material/color/etc. below -- a stated category is just as real a
    # signal on the Browsing track as on Buying, and it's the one attribute virtually every real
    # session has from turn 1 (see stated_hard_slots()'s docstring for why this needed its own
    # flag rather than joining ATTRIBUTE_BUCKET_FUNCTIONS's track=="buying" gate).
    if attribute == "category" and CATEGORY_HARD_MATCH_ENABLED:
        return _bucket_match(CATEGORY_BUCKET_FUNCTION, raw_value, product)
    if track == "buying" and attribute in ATTRIBUTE_BUCKET_FUNCTIONS:
        return _bucket_match(ATTRIBUTE_BUCKET_FUNCTIONS[attribute], raw_value, product)
    return match_text_generic(raw_value, product)


def stated_hard_slots(state: SessionState) -> list[tuple[str, str]]:
    """Every consumer of this function (guarantee_pass.py's boost/demote, rank.py's soft_slot
    score, precision_pass.category_rescue's violate-check, evaluate_domain() below) shares
    whatever this returns -- category was excluded here unconditionally until round 17, which
    meant NONE of those four ever checked that a candidate was even the right kind of product
    (see CATEGORY_HARD_MATCH_ENABLED in config/thresholds.py for the live-testing finding that
    surfaced this and why it's flagged rather than unconditional).
    """
    return [
        (attribute, slot.value)
        for attribute, slot in state.slots.items()
        if (attribute != "category" or CATEGORY_HARD_MATCH_ENABLED)
        and slot.status in ("stated", "overridden")
        and slot.value
    ]


@dataclass
class DomainSignal:
    size: int
    over_generality: bool
    collision: bool


def evaluate_domain(candidate_asins: list[str], products: dict[str, dict], state: SessionState) -> DomainSignal:
    pool_size = len(candidate_asins)
    stated = stated_hard_slots(state)
    if not stated:
        return DomainSignal(size=pool_size, over_generality=pool_size > OVER_GENERALITY_DOMAIN_SIZE, collision=False)

    satisfying = [
        asin
        for asin in candidate_asins
        if all(match_text(attribute, value, products[asin], state.track) != "violate" for attribute, value in stated)
    ]
    size = len(satisfying)
    return DomainSignal(
        size=size,
        over_generality=size > OVER_GENERALITY_DOMAIN_SIZE,
        collision=size < COLLISION_DOMAIN_SIZE and pool_size >= COLLISION_DOMAIN_SIZE,
    )

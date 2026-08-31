"""Text-parsing helpers that scoutie's own understanding/retrieval modules depend on.

Round 14, Tier 0a (packaging-independence, not a style preference): these were previously imported
directly from `starter.agent` and `evaluator.local_evaluator` -- organizer-provided reference/
grading modules, not scoutie's own code. That's a real risk, not a cosmetic one: those imports run
at module-import time, before `Agent()` is ever constructed. If the official private-set harness
packages or runs the submission with a different working directory or file layout than this local
repo (e.g. only `scoutie/` is copied), every one of those imports fails at import time and
`Agent()` can never be constructed -- a total zero for the whole submission, not a degraded turn.
See CLAUDE.md and PROGRESS.md round 14 for the full rationale.

Everything below is a verbatim relocation, not a rewrite -- copied byte-for-behavior identical from
its original source, with no logic changes. Each block below cites exactly where it came from.

Round 17 exception, noted here so this claim stays accurate: classify_constraint()'s use_case
keyword tuple gained a few real-world synonyms (see the constant's own comment below) after a live
dashboard session showed "should be for office wear" falling through to the generic "feature"
bucket, which then caused strategy.py to re-ask use_case later even though the shopper had already
answered it in different words. This is now a deliberate, evidence-motivated divergence from the
evaluator's original vocabulary, not a byte-for-behavior copy -- verified against the full
evaluator before being kept (see PROGRESS.md round 17).
"""

from __future__ import annotations

import re

# --- Copied verbatim from starter/agent.py (organizer-provided starter kit) ---

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


# --- Copied verbatim from evaluator/local_evaluator.py (organizer-provided local evaluator) ---

MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)


# Round 17 addition (not in the evaluator's original list -- see module docstring): "office",
# "business", "professional", "formal", "workplace" cover common real-world use-case phrasing
# ("for office wear", "for the office") that the original list's single "work" entry missed,
# since "office wear" doesn't contain the literal substring "work".
USE_CASE_KEYWORDS_EXTENDED = (
    "hiking", "running", "gym", "winter", "outdoor", "work",
    "office", "business", "professional", "formal", "workplace",
)

# Round 18c addition: the original list only detects the VOCABULARY of talking about size
# ("size", "sizing", "width"...), not an actual standalone size VALUE. Confirmed live: a bare
# value like "medium" -- exactly what the dashboard's quick-reply size buttons send as raw chat
# text (see dashboard/api.py's _quick_replies_for()), no surrounding "size" word attached -- fell
# through to "feature" instead of "size" without this. No bare single letters (S/M/L) here -- "m"
# or "l" as a substring check would false-positive against nearly anything; only unambiguous size
# words are safe additions.
SIZE_VALUE_KEYWORDS_EXTENDED = (
    "size", "sizing", "width", "wide", "narrow",
    "small", "medium", "large", "petite", "plus size",
    "xs", "xl", "xxl", "x-small", "extra small", "extra large",
)


def classify_constraint(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(word in lowered for word in SIZE_VALUE_KEYWORDS_EXTENDED):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in USE_CASE_KEYWORDS_EXTENDED):
        return "use_case"
    return "feature"

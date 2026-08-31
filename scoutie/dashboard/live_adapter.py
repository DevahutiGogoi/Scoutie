"""Bridges a real human's free-typed chat message to the scripted phrasing
understanding/belief_revision.py actually parses.

Why this exists: belief_revision.apply_customer_reply() and parse_turn_one() are regex-matched
against evaluator/local_evaluator.py's own customer_reply() templates verbatim (e.g. "For that,
what matters is: X.", "I don't have a preference for X; please use your judgment."). That's
correct and deliberate for the scored evaluator path -- but it means a real person typing "black,
under $40" into the live chat UI would fall through every regex and hit apply_customer_reply()'s
last-resort fallback, which only fires when the asked slot is already "stated" (i.e. it can
revise an answer but cannot record a first answer at all). Confirmed by reading
belief_revision.py directly, not assumed.

This module does not replace or extend belief_revision.py's own logic -- it never classifies a
constraint or decides a status itself. It only re-wraps a real message into the exact literal
scripts belief_revision.py already knows how to parse, then hands that wrapped string to
agent.respond() unchanged. All actual extraction/classification still happens inside scoutie's
real, tested production code path.

Round 19 note: an earlier version of this module was optionally assisted by a small local model
(local_llm_extractor.py) for splitting a free sentence into multiple facts at once. That module
was removed once the deployed target (AWS free tier, 1GB RAM) couldn't fit it and quick-reply
chips (dashboard/api.py) covered most of what it bought in practice -- see PROGRESS.md for the
full account. This module was accordingly reverted to the pure-regex version below; it never
depended on the model being present, so nothing here changed structurally.
"""

from __future__ import annotations

import re

from scoutie.understanding.belief_revision import CATEGORY_RE, DECLINE_KEYWORDS

_TRAILING_PUNCT_RE = re.compile(r"[.\s]+$")

# A budget-shaped clause embedded in an otherwise-category turn-1 message (e.g. "jacket under
# $150"). Matched separately from the rest of the sentence so turn 1 can produce belief_revision's
# TWO real scripted fragments ("I'm looking for jacket." + "A key requirement is: under $150.")
# instead of swallowing the whole sentence into the category slot alone -- confirmed live: without
# this split, a live session's stated $150 budget was silently dropped (never stored anywhere,
# not even overwritten later) because parse_turn_one()'s CATEGORY_RE captured the entire message,
# leaving FREE_HARD_CONSTRAINT_RE nothing to match.
_BUDGET_CLAUSE_RE = re.compile(
    r"(?:,?\s*(?:for\s+)?(?:under|below|less than|no more than|up to)\s*\$?\d+(?:\.\d+)?|\$\d+(?:\.\d+)?)",
    re.I,
)


def _clean(raw: str) -> str:
    return _TRAILING_PUNCT_RE.sub("", raw.strip())


def _split_turn_one(cleaned: str) -> tuple[str, str | None]:
    """cleaned -> (category_text, budget_clause_or_None). Only ever pulls out a budget-shaped
    clause -- material/color/etc. mixed into turn 1 still land nowhere until asked about; solving
    that in general is a full NLU problem, not a bounded regex split."""
    match = _BUDGET_CLAUSE_RE.search(cleaned)
    if not match:
        return cleaned, None
    budget_clause = match.group(0).strip(" ,")
    remainder = (cleaned[: match.start()] + cleaned[match.end() :]).strip(" ,")
    return (remainder or cleaned), budget_clause


def adapt_message(raw_message: str, turn: int, last_asked_attribute: str | None) -> str:
    """Returns the literal string to hand to agent.respond() in place of the user's raw text.

    turn: the turn number about to be sent (1 for the first message of the session).
    last_asked_attribute: state.strategy_history[-1] from the PREVIOUS turn, or None if this is
    turn 1 or nothing was asked yet.
    """
    cleaned = _clean(raw_message)
    if not cleaned:
        return raw_message

    lowered = cleaned.lower()
    is_decline = any(keyword in lowered for keyword in DECLINE_KEYWORDS)

    if turn == 1:
        if CATEGORY_RE.search(raw_message):
            return raw_message
        category_text, budget_clause = _split_turn_one(cleaned)
        if budget_clause:
            return f"I'm looking for {category_text}. A key requirement is: {budget_clause}."
        return f"I'm looking for {cleaned}."

    if is_decline and last_asked_attribute and last_asked_attribute != "other":
        return f"I don't have a preference for {last_asked_attribute}; please use your judgment."

    # Every other live reply -- a first answer, a revised answer, or an "other"-prompted answer --
    # goes through DISCLOSURE_RE, exactly like the evaluator's own customer_reply() output.
    # apply_customer_reply() already handles "already stated + different value" as an override on
    # its own (see belief_revision.py), so no separate override-phrase synthesis is needed here.
    return f"For that, what matters is: {cleaned}."

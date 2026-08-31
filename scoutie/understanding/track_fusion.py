"""Buying vs. Browsing decision via Dempster-Shafer evidence combination across three
independent belief masses: the message signal, the profile prior, and a conviction
(hedge-vs-certainty) signal. Track becomes "buying" once accumulated belief crosses
TRACK_FUSION_BUYING_BELIEF_THRESHOLD, at which point it ratchets permanently -- it never
reverts to "browsing" once locked, since a hard constraint disclosed under a Buying track
should never be un-learned by later vaguer phrasing.

Hedge/certainty word lists live in two tiers in config/thresholds.py: AUDITED (built from
directly scanning the customer simulator's own generated text) and SUPPLEMENTARY (broader
coverage for phrasing the simulator doesn't currently use, kept for private-set robustness).
_strip_templated_value() below keeps catalog text quoted verbatim into a disclosure/override
reply (e.g. "100% Cotton") out of the hedge/certainty scan, so a fabric composition string is
never misread as customer conviction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scoutie.config.thresholds import (
    BUYING_RATCHET,
    CERTAINTY_MARKERS_AUDITED,
    CERTAINTY_MARKERS_SUPPLEMENTARY,
    CONVICTION_MASS_STRENGTH,
    EXPLORATORY_KEYWORDS,
    EXPLORATORY_KEYWORDS_SUPPLEMENTARY,
    HEDGE_MARKERS_AUDITED,
    HEDGE_MARKERS_SUPPLEMENTARY,
    MESSAGE_MASS_BROWSING_STRENGTH,
    MESSAGE_MASS_BUYING_STRENGTH,
    MESSAGE_MASS_CONFLICT_SPLIT,
    PROFILE_MASS_STRENGTH,
    TRACK_FUSION_BROWSING_BELIEF_THRESHOLD,
    TRACK_FUSION_BUYING_BELIEF_THRESHOLD,
    TRANSACTIONAL_KEYWORDS,
    TRANSACTIONAL_KEYWORDS_SUPPLEMENTARY,
)
from scoutie.state import SessionState
from scoutie.understanding import belief_revision


# --- Dempster-Shafer combination over Theta = {buying, browsing} ---


@dataclass(frozen=True)
class Mass:
    """A mass function over {buying}, {browsing}, and Theta ({buying, browsing} -- ignorance).
    The empty set always gets zero mass by construction; conflict between two mass functions
    (mass that would land on the empty set) is computed and normalized away in dempster_combine().
    """

    buying: float = 0.0
    browsing: float = 0.0
    uncertain: float = 1.0


def dempster_combine(m1: Mass, m2: Mass) -> Mass:
    """Dempster's rule of combination for two independent mass functions over a 2-element frame.
    Combining with Mass() (pure ignorance) is an identity operation -- this is what lets a turn
    with no evidence leave the running belief untouched, and what makes a single fixed prior safe
    to combine in only once (see classify_track_dempster_shafer()).
    """
    conflict = m1.buying * m2.browsing + m1.browsing * m2.buying
    if conflict >= 1.0 - 1e-9:
        # Total conflict (e.g. one source pins "buying", the other pins "browsing"): treat as
        # pure ignorance rather than divide by ~zero. Doesn't arise with the strengths in
        # thresholds.py (no source ever assigns 1.0 to a singleton) but guarded regardless.
        return Mass(0.0, 0.0, 1.0)
    normalizer = 1.0 - conflict
    buying = (m1.buying * m2.buying + m1.buying * m2.uncertain + m1.uncertain * m2.buying) / normalizer
    browsing = (m1.browsing * m2.browsing + m1.browsing * m2.uncertain + m1.uncertain * m2.browsing) / normalizer
    uncertain = (m1.uncertain * m2.uncertain) / normalizer
    return Mass(buying, browsing, uncertain)


def _mass_to_dict(mass: Mass) -> dict:
    return {"buying": mass.buying, "browsing": mass.browsing, "uncertain": mass.uncertain}


def _mass_from_dict(data: dict) -> Mass:
    return Mass(data.get("buying", 0.0), data.get("browsing", 0.0), data.get("uncertain", 1.0))


def _strip_templated_value(message: str) -> str:
    """Returns the customer simulator's own scaffold text, with any catalog value it quoted
    verbatim stripped out. Reuses belief_revision.py's own regexes (the same patterns that parse
    these templates for slot extraction) rather than duplicating pattern logic. Falls back to the
    whole message unchanged when nothing recognized matches -- the case that matters for a
    private-set message that paraphrases instead of following the scripted templates.
    """
    for pattern in (belief_revision.OVERRIDE_RE, belief_revision.DISCLOSURE_RE):
        match = pattern.match(message)
        if match:
            return message[: match.start(1)]
    match = belief_revision.FREE_HARD_CONSTRAINT_RE.search(message)
    if match:
        return message[: match.start(1)]
    return message


def detect_conviction(message: str) -> float:
    """Hedge-vs-certainty score: -1.0 (hedged), 0.0 (neither, or both at once -- genuinely
    ambiguous), or 1.0 (certain). Always strips template-quoted catalog text internally first
    (safe to call with either a raw or an already-stripped message -- stripping is idempotent).

    Deliberately does not fire on Boundary-decline phrasing ("no preference", "use your
    judgment") even though a couple of hedge markers overlap textually (e.g. "flexible on") --
    that path is a separate detector (belief_revision.BOUNDARY_DECLINE_RE / DECLINE_KEYWORDS)
    and a decline should register as a resolved slot, not merely a hedge.
    """
    if belief_revision.BOUNDARY_DECLINE_RE.match(message):
        return 0.0
    lowered = _strip_templated_value(message).lower()
    if any(keyword in lowered for keyword in belief_revision.DECLINE_KEYWORDS):
        return 0.0
    has_certainty = any(marker in lowered for marker in CERTAINTY_MARKERS_AUDITED + CERTAINTY_MARKERS_SUPPLEMENTARY)
    has_hedge = any(marker in lowered for marker in HEDGE_MARKERS_AUDITED + HEDGE_MARKERS_SUPPLEMENTARY)
    if has_certainty and has_hedge:
        return 0.0
    if has_certainty:
        return 1.0
    if has_hedge:
        return -1.0
    return 0.0


def _message_mass(clean_text: str, just_disclosed_hard_slot: bool) -> Mass:
    lowered = clean_text.lower()
    has_transactional = any(kw in lowered for kw in TRANSACTIONAL_KEYWORDS + TRANSACTIONAL_KEYWORDS_SUPPLEMENTARY)
    has_exploratory = any(kw in lowered for kw in EXPLORATORY_KEYWORDS + EXPLORATORY_KEYWORDS_SUPPLEMENTARY)
    buying_evidence = has_transactional or just_disclosed_hard_slot
    browsing_evidence = has_exploratory

    if buying_evidence and browsing_evidence:
        split = MESSAGE_MASS_CONFLICT_SPLIT
        return Mass(split, split, 1.0 - 2 * split)
    if buying_evidence:
        return Mass(MESSAGE_MASS_BUYING_STRENGTH, 0.0, 1.0 - MESSAGE_MASS_BUYING_STRENGTH)
    if browsing_evidence:
        return Mass(0.0, MESSAGE_MASS_BROWSING_STRENGTH, 1.0 - MESSAGE_MASS_BROWSING_STRENGTH)
    return Mass()


def _conviction_mass(clean_text: str) -> Mass:
    score = detect_conviction(clean_text)
    if score > 0:
        return Mass(CONVICTION_MASS_STRENGTH, 0.0, 1.0 - CONVICTION_MASS_STRENGTH)
    if score < 0:
        return Mass(0.0, CONVICTION_MASS_STRENGTH, 1.0 - CONVICTION_MASS_STRENGTH)
    return Mass()


def _profile_mass(profile_prior: dict) -> Mass:
    """purchase_frequency is a free-text field (e.g. "3-4 prior purchases"); any leading count
    gets a deliberately mild lean toward Buying, otherwise pure ignorance."""
    frequency_text = str(profile_prior.get("purchase_frequency") or "")
    count_match = re.search(r"\d+", frequency_text)
    if count_match and int(count_match.group()) >= 1:
        return Mass(PROFILE_MASS_STRENGTH, 0.0, 1.0 - PROFILE_MASS_STRENGTH)
    return Mass()


def classify_track_dempster_shafer(message: str, state: SessionState) -> str:
    """Combines the message-signal and conviction masses for this turn, folds in the
    (session-constant) profile-prior mass exactly once -- the first time any evidence is
    combined, not every turn, since repeatedly re-combining an unchanging mass would silently
    accumulate and saturate belief toward its favored side purely from the passage of turns,
    independent of anything actually said -- then combines that turn's total evidence into the
    running session belief (state.track_belief, persisted across turns). Track becomes "buying"
    only once the accumulated pignistic belief crosses TRACK_FUSION_BUYING_BELIEF_THRESHOLD, at
    which point the ratchet locks it permanently.
    """
    if state.track == "buying" and BUYING_RATCHET:
        return "buying"

    clean_text = _strip_templated_value(message)
    just_disclosed_hard_slot = any(
        slot.status in ("stated", "overridden") and slot.last_updated_turn == state.turn_count
        for attribute, slot in state.slots.items()
        if attribute != "category"
    )

    turn_evidence = dempster_combine(
        _message_mass(clean_text, just_disclosed_hard_slot),
        _conviction_mass(clean_text),
    )
    if not state.track_profile_mass_applied:
        turn_evidence = dempster_combine(turn_evidence, _profile_mass(state.profile_prior))
        state.track_profile_mass_applied = True

    running = _mass_from_dict(state.track_belief)
    combined = dempster_combine(running, turn_evidence)
    state.track_belief = _mass_to_dict(combined)

    buying_belief = combined.buying + 0.5 * combined.uncertain
    browsing_belief = combined.browsing + 0.5 * combined.uncertain
    if buying_belief >= TRACK_FUSION_BUYING_BELIEF_THRESHOLD:
        return "buying"
    if browsing_belief >= TRACK_FUSION_BROWSING_BELIEF_THRESHOLD:
        return "browsing"
    return state.track if state.track != "unknown" else "browsing"

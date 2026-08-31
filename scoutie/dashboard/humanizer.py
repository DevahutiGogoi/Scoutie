"""Option A humanizing layer: template-based rephrasing of an already-fully-decided message.

Zero model inference. This module never decides what to recommend, which attribute to ask, or
whether a slot got a preference/override/decline -- all of that is already fixed by the time this
runs (agent.respond()'s real return dict, plus the live SessionState behind it). This module's
only job is picking a natural-sounding surface form for content that's already decided, from a
fixed set of hand-written templates with light variation.

Deliberately lives in scoutie/dashboard/, not scoutie/ core -- this is a live-product-UI display
concern, not part of the evaluator-scored respond() path. agent.py's own `message` field (scored
by nothing but "is it a non-empty string", per CLAUDE.md) is never touched or read by this module;
the dashboard API calls this separately, after respond(), using the real SessionState it already
has for other panels.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

ATTRIBUTE_PHRASES = {
    "material": "material",
    "color": "color",
    "size": "size",
    "style": "style",
    "budget": "budget",
    "use_case": "what you'll be using it for",
    "feature": "what matters most to you",
}

# --- Acknowledgment fragments: lead-in clause before this turn's real content. ---

PLAIN_ACK_TEMPLATES = (
    "Got it.",
    "Thanks, noted.",
    "Good to know.",
    "Makes sense.",
    "Noted, thanks.",
    "Okay, got that.",
)

# Referencing the specific attribute+value just captured, when we have both (see
# TurnEvent.just_captured below). Falls back to PLAIN_ACK_TEMPLATES when we don't -- e.g. the
# "other" broad ask can capture a value under an attribute the UI has no short human phrase for.
PLAIN_ACK_WITH_VALUE_TEMPLATES = (
    "Got it -- {value}.",
    "Noted -- {value}.",
    "Got it, {value} it is.",
    "Good to know -- {value}.",
)

ASK_OTHER_WITH_GAPS_TEMPLATES = (
    "What about {gaps} -- anything I should know?",
    "Anything on {gaps} that matters to you?",
    "Is there anything about {gaps} I should factor in?",
)

OVERRIDE_ACK_TEMPLATES = (
    "Got it, updating that.",
    "No problem, switching that up.",
    "Sure thing -- I'll go with that instead.",
    "Updated, let's go from there.",
    "Got it, changing course on that one.",
    "Understood -- I'll swap that in.",
)

DECLINE_ACK_TEMPLATES = (
    "No worries, I'll use my best judgment there.",
    "Sure, I'll figure that part out myself.",
    "Fair enough -- I'll take it from here on that one.",
    "That's okay, I'll make the call on that.",
    "Got it, I won't worry about that one.",
)

EXHAUSTED_ACK_TEMPLATES = (
    "Alright, nothing more to add there.",
    "Okay, that's everything on that front.",
    "Understood, moving on.",
)

# --- Clarifying questions: asking about one named attribute. ---

ASK_ATTRIBUTE_TEMPLATES = (
    "To help narrow this down, what's your take on {attr}?",
    "One more thing -- any preference on {attr}?",
    "Could you tell me a bit about the {attr} you're after?",
    "What about {attr} -- anything specific in mind?",
    "Mind sharing your preference on {attr}?",
    "Quick question -- what are you thinking for {attr}?",
)

# --- Clarifying questions: the broad "other" ask, no single attribute named. ---

ASK_OTHER_TEMPLATES = (
    "Is there anything else about what you're looking for that I should know?",
    "Anything else important I should factor in?",
    "What else matters to you here?",
    "Tell me one more thing that would help me narrow this down.",
    "Anything you haven't mentioned yet that I should keep in mind?",
    "What other details matter for this one?",
)

# --- Best-effort recommendation, no question this turn, plain case. ---

RECOMMENDATION_TEMPLATES = (
    "Here's what's looking like a good fit so far.",
    "Here are my top picks right now.",
    "These are standing out as strong matches.",
    "Take a look at these -- they're my best guesses so far.",
    "Here's where things stand -- these look promising.",
    "These are the closest matches I've got for you.",
)

# --- Best-effort recommendation, pool is still wide open (over-generality). ---

OVER_GENERALITY_TEMPLATES = (
    "There's a lot to choose from here -- here's a first pass while I narrow things down.",
    "Plenty of options match so far -- here's a starting shortlist.",
    "Still quite a few possibilities -- here's where things stand.",
    "Lots to work with still -- here are some options to react to.",
    "There's a wide range that fits so far -- here's a sample to start narrowing from.",
)

# --- Best-effort recommendation, constraints are colliding / pool collapsed. ---

COLLISION_TEMPLATES = (
    "I'm having a little trouble matching everything you've told me exactly -- here's my best-effort list.",
    "A couple of your preferences seem to be pulling in different directions, so here's my closest guess.",
    "Nothing fits every detail perfectly, so here's the nearest match I could find.",
    "It's a tight fit given everything you've said -- here's what comes closest.",
)

# --- Session close: user marked a recommendation as the one. ---

SESSION_CLOSE_TEMPLATES = (
    "Great choice -- glad that worked out!",
    "Nice, that one's a keeper.",
    "Love it -- hope it's exactly what you needed.",
    "Good pick! Glad I could help track that down.",
    "That's a great find -- enjoy!",
)


@dataclass
class TurnEvent:
    """What actually happened this turn, read straight off the real SessionState -- never
    inferred or guessed here. Built by dashboard/api.py from state.strategy_history,
    state.override_events, and the slot statuses, before calling humanize()."""

    turn_count: int
    ask_attribute: str | None
    over_generality: bool
    collision: bool
    just_overrode: bool  # an override_events entry exists for this exact turn
    just_declined: bool  # the previously-asked slot's status became "no_preference" this turn
    just_exhausted: bool  # the previously-asked slot's status became "exhausted" this turn
    had_prior_ask: bool  # a real (non-"other", non-None) attribute was asked last turn
    converted: bool = False
    # (attribute, value) just captured this turn from a plain (non-override/decline/exhausted)
    # answer, if there is one worth naming -- lets the acknowledgment reference what was actually
    # learned ("Got it -- navy.") instead of a content-free "Got it." every time. None when there's
    # nothing worth naming (turn 1, or the captured value has no short human-readable form).
    just_captured_value: str | None = None
    # Askable attributes still "unknown" as of this turn, human-readable (e.g. "budget", "what
    # you'll use it for") -- used to make the broad "other" ask name what's actually still missing
    # instead of a fully content-free "anything else?" every time.
    unknown_attribute_phrases: tuple[str, ...] = ()


def _pick(templates: tuple[str, ...]) -> str:
    return random.choice(templates)


def _lead_in(event: TurnEvent) -> str | None:
    if event.turn_count <= 1:
        return None
    if event.just_overrode:
        return _pick(OVERRIDE_ACK_TEMPLATES)
    if event.just_declined:
        return _pick(DECLINE_ACK_TEMPLATES)
    if event.just_exhausted:
        return _pick(EXHAUSTED_ACK_TEMPLATES)
    if event.just_captured_value:
        return _pick(PLAIN_ACK_WITH_VALUE_TEMPLATES).format(value=event.just_captured_value)
    if event.had_prior_ask:
        return _pick(PLAIN_ACK_TEMPLATES)
    return None


def _body(event: TurnEvent) -> str:
    if event.ask_attribute and event.ask_attribute != "other":
        phrase = ATTRIBUTE_PHRASES.get(event.ask_attribute, event.ask_attribute.replace("_", " "))
        return _pick(ASK_ATTRIBUTE_TEMPLATES).format(attr=phrase)
    if event.ask_attribute == "other":
        # Name 1-2 of what's actually still missing rather than a fully content-free "anything
        # else?" every time -- strategy.py's own decision to ask broadly (see
        # ASK_OTHER_MAX_COUNT in config/thresholds.py) is unchanged; this only affects which
        # words describe that same ask.
        if event.unknown_attribute_phrases:
            gaps = " or ".join(event.unknown_attribute_phrases[:2])
            return _pick(ASK_OTHER_WITH_GAPS_TEMPLATES).format(gaps=gaps)
        return _pick(ASK_OTHER_TEMPLATES)
    if event.collision:
        return _pick(COLLISION_TEMPLATES)
    if event.over_generality:
        return _pick(OVER_GENERALITY_TEMPLATES)
    return _pick(RECOMMENDATION_TEMPLATES)


def humanize(event: TurnEvent) -> str:
    """The only entry point. Never called on the scored respond() path -- see module docstring."""
    if event.converted:
        return _pick(SESSION_CLOSE_TEMPLATES)
    lead = _lead_in(event)
    body = _body(event)
    return f"{lead} {body}" if lead else body

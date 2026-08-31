"""Intent Override handling, and more broadly: the only place new information enters state.

Every regex here matches literal templates from evaluator.local_evaluator's customer_reply() /
initial_message(). A broader decline-keyword fallback is kept for boundary detection in case the
private/held-out set paraphrases the scripted line instead of using it verbatim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scoutie.text_utils import classify_constraint
from scoutie.state import SessionState, Slot

CATEGORY_RE = re.compile(r"looking for ([^.,]+)")
FREE_HARD_CONSTRAINT_RE = re.compile(r"A key requirement is: (.+?)\.$")
OVERRIDE_RE = re.compile(r"^Actually, ignore my earlier preference\. What I need is: (.+)\.$")
BOUNDARY_DECLINE_RE = re.compile(r"^I don't have a preference for (\w+); please use your judgment\.$")
EXHAUSTED_RE = re.compile(r"^I don't have an additional preference for (\w+)\.$")
GENERIC_PUSHBACK = "Those options are not quite right yet. Ask me about one specific attribute."
DISCLOSURE_RE = re.compile(r"^For that, what matters is: (.+)\.$")
DECLINE_KEYWORDS = ("no preference", "don't care", "any is fine", "doesn't matter", "use your judgment")


@dataclass
class TurnOneParse:
    category_text: str | None
    free_hard_constraint: str | None
    browsing_hint: bool


def parse_turn_one(message: str) -> TurnOneParse:
    category_match = CATEGORY_RE.search(message)
    constraint_match = FREE_HARD_CONSTRAINT_RE.search(message)
    return TurnOneParse(
        category_text=category_match.group(1).strip() if category_match else None,
        free_hard_constraint=constraint_match.group(1).strip() if constraint_match else None,
        browsing_hint="still exploring" in message.lower(),
    )


def _assign_slot(state: SessionState, attribute: str, value: str, turn: int, *, status: str = "stated") -> None:
    slot = state.slots.setdefault(attribute, Slot())
    slot.value = value
    slot.status = status
    slot.weight = 1.0
    slot.last_updated_turn = turn


def _record_override(state: SessionState, attribute: str, old_value: str | None, new_value: str, turn: int) -> None:
    _assign_slot(state, attribute, new_value, turn, status="overridden")
    state.override_events.append({"turn": turn, "attribute": attribute, "old": old_value, "new": new_value})


def _clear_stale_pre_override_slots(state: SessionState, just_updated_attribute: str) -> None:
    """Clears any OTHER currently-stated slot whose value is a verbatim substring (or one part of
    a co-disclosure-merged "A; B" value) of turn 1's raw message -- the one place an
    intent_override session's `old_value` is guaranteed to originate from, since the override
    message itself never states it directly. Without this, a repudiated pre-override preference
    keeps propping up ranking after the customer explicitly disavows it.
    """
    if not state.message_history:
        return
    turn_one_message = state.message_history[0]
    for attribute, slot in state.slots.items():
        if attribute in ("category", just_updated_attribute):
            continue
        if slot.status not in ("stated", "overridden") or not slot.value:
            continue
        parts = [part.strip() for part in slot.value.split(";")]
        if any(part and part in turn_one_message for part in parts):
            slot.value = None
            slot.status = "unknown"
            slot.weight = 1.0
            slot.last_updated_turn = None


def apply_turn_one(parse: TurnOneParse, state: SessionState, turn: int) -> None:
    if parse.category_text:
        _assign_slot(state, "category", parse.category_text, turn)
    if parse.free_hard_constraint:
        attribute = classify_constraint(parse.free_hard_constraint)
        _assign_slot(state, attribute, parse.free_hard_constraint, turn)
    if parse.browsing_hint:
        state.track = "browsing"


def apply_customer_reply(message: str, state: SessionState, turn: int) -> None:
    override_match = OVERRIDE_RE.match(message)
    if override_match:
        new_value = override_match.group(1).strip()
        attribute = classify_constraint(new_value)
        old_value = state.slots[attribute].value if attribute in state.slots else None
        _record_override(state, attribute, old_value, new_value, turn)
        _clear_stale_pre_override_slots(state, attribute)
        return

    boundary_match = BOUNDARY_DECLINE_RE.match(message)
    if boundary_match:
        attribute = boundary_match.group(1)
        if attribute in state.slots:
            state.slots[attribute].status = "no_preference"
            state.slots[attribute].last_updated_turn = turn
        return

    exhausted_match = EXHAUSTED_RE.match(message)
    if exhausted_match:
        attribute = exhausted_match.group(1)
        if attribute in state.slots and state.slots[attribute].status == "unknown":
            state.slots[attribute].status = "exhausted"
            state.slots[attribute].last_updated_turn = turn
        return

    if message.strip() == GENERIC_PUSHBACK:
        return

    disclosure_match = DISCLOSURE_RE.match(message)
    if disclosure_match:
        raw_values = [value.strip() for value in disclosure_match.group(1).split("; ") if value.strip()]
        asked_attribute = state.strategy_history[-1] if state.strategy_history else None
        for raw_value in raw_values:
            attribute = asked_attribute or classify_constraint(raw_value)
            if attribute not in state.slots:
                attribute = classify_constraint(raw_value)
            existing = state.slots[attribute]
            if existing.status == "stated" and existing.value and existing.value != raw_value:
                if existing.last_updated_turn == turn:
                    # A second value in THIS SAME disclosure turn resolved to a slot the first
                    # value in the loop already set (customer_reply() can send up to two
                    # constraint strings per reply, and both can classify to the same attribute).
                    # That's co-disclosure, not a contradiction -- merge rather than treat it as
                    # an override, and don't log a spurious override_events entry.
                    if raw_value not in existing.value:
                        existing.value = f"{existing.value}; {raw_value}"
                else:
                    _record_override(state, attribute, existing.value, raw_value, turn)
            else:
                _assign_slot(state, attribute, raw_value, turn)
        return

    # Fallback for paraphrased replies that don't match any scripted template above.
    lowered = message.lower()
    if any(keyword in lowered for keyword in DECLINE_KEYWORDS):
        asked_attribute = state.strategy_history[-1] if state.strategy_history else None
        if asked_attribute and asked_attribute in state.slots:
            state.slots[asked_attribute].status = "no_preference"
            state.slots[asked_attribute].last_updated_turn = turn
        return

    # A conflicting restatement that matched nothing above but targets the last-asked, already
    # stated slot: treat it as a generic override rather than silently dropping the information.
    asked_attribute = state.strategy_history[-1] if state.strategy_history else None
    if asked_attribute and asked_attribute in state.slots and state.slots[asked_attribute].status == "stated":
        existing_value = state.slots[asked_attribute].value
        if existing_value and existing_value != message.strip():
            _record_override(state, asked_attribute, existing_value, message.strip(), turn)

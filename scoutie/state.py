"""Per-session state: the single source of truth mutated once per turn before anything else runs."""

from __future__ import annotations

from dataclasses import dataclass, field

from scoutie.config.thresholds import SLOT_ATTRIBUTES


@dataclass
class Slot:
    value: str | None = None
    weight: float = 1.0
    status: str = "unknown"  # unknown | stated | no_preference | exhausted | overridden
    last_updated_turn: int | None = None


@dataclass
class SessionState:
    session_id: str
    turn_count: int = 0
    track: str = "unknown"  # unknown | browsing | buying
    profile_prior: dict = field(default_factory=dict)
    slots: dict[str, Slot] = field(default_factory=dict)
    strategy_history: list[str] = field(default_factory=list)
    override_events: list[dict] = field(default_factory=list)
    message_history: list[str] = field(default_factory=list)
    # Dempster-Shafer running belief over {buying, browsing} for track_fusion.py. Plain dict, not
    # a dataclass, to avoid a state.py <-> track_fusion.py import cycle -- track_fusion.py
    # converts to/from its own Mass dataclass internally. Starts at total ignorance
    # (uncertain=1.0): no evidence yet.
    track_belief: dict = field(default_factory=lambda: {"buying": 0.0, "browsing": 0.0, "uncertain": 1.0})
    # The profile prior is constant for the whole session (seeded once at reset()) -- this flags
    # that it's already been folded into track_belief once, so it isn't re-combined every turn
    # (which would let a constant, non-decaying mass silently accumulate and saturate belief
    # toward its favored side over a long session regardless of what's actually being said).
    track_profile_mass_applied: bool = False
    # How many times "other" has been asked this session (customer_reply()'s match filter
    # collapses to "any not-yet-disclosed constraint, any type" when the asked attribute is
    # "other" -- see orchestration/strategy.py). "other" has no real state.slots key, so its use
    # is tracked here rather than via slot status like every other askable attribute.
    other_ask_count: int = 0
    # Last turn's constraint_engine.DomainSignal, unpacked into plain fields (not the dataclass
    # itself -- constraint_engine.py imports SessionState from this module, so importing
    # DomainSignal back here would cycle; same pattern as track_belief above). Read-only, for the
    # dashboard API's unscored debug endpoint -- never read by anything in the scored respond()
    # path itself.
    last_pool_size: int = 0
    last_over_generality: bool = False
    last_collision: bool = False
    # The actual candidate ASINs behind last_pool_size, for the dashboard's quick-reply chips
    # (dashboard/api.py) to compute real per-attribute distinct values from -- same read-only,
    # UI-only status as the three fields above.
    last_candidate_asins: list[str] = field(default_factory=list)


def new_session(session_id: str, user_profile: dict) -> SessionState:
    return SessionState(
        session_id=session_id,
        profile_prior=dict(user_profile or {}),
        slots={attribute: Slot() for attribute in SLOT_ATTRIBUTES},
    )

"""Thin orchestrator: wires the modules under scoutie/ together. No real logic lives here."""

from __future__ import annotations

from pathlib import Path

from scoutie.config.thresholds import (
    CATEGORY_RESCUE_COUNT,
    CATEGORY_RESCUE_PICK_ZONE,
    CATEGORY_RESCUE_TOP_K,
    DEFAULT_MESSAGE,
    DISTILLATION_CONTEXT_TURNS,
    FUSION_TOP_N_PER_ROUTE,
    PRECISION_PASS_TOP_K,
    ROUTE_WEIGHTS_BROWSING,
    ROUTE_WEIGHTS_BUYING,
)
from scoutie import strategy
from scoutie.ranking import guarantee_pass, rank
from scoutie.retrieval import fusion, precision_pass
from scoutie.retrieval.routes import Routes
from scoutie.state import SessionState, new_session
from scoutie.understanding import belief_revision, constraint_engine, track_fusion

ALLOWED_ASK_ATTRIBUTES = {
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other",
}


def _build_query_text(state: SessionState) -> str:
    slot_values = [
        slot.value
        for attribute, slot in state.slots.items()
        if slot.value and slot.status in ("stated", "overridden")
    ]
    context_text = " ".join(state.message_history[-DISTILLATION_CONTEXT_TURNS:])
    return " ".join([*slot_values, context_text])


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.routes = Routes(catalog_path)
        self.products: dict[str, dict] = dict(zip(self.routes.doc_ids, self.routes.rows))
        self.sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = new_session(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            state = self.sessions.setdefault(session_id, new_session(session_id, {}))
            state.turn_count = turn
            state.message_history.append(user_message)

            if turn == 1:
                parse = belief_revision.parse_turn_one(user_message)
                belief_revision.apply_turn_one(parse, state, turn)
            else:
                belief_revision.apply_customer_reply(user_message, state, turn)

            state.track = track_fusion.classify_track_dempster_shafer(user_message, state)

            query_text = _build_query_text(state)
            category_hint = state.slots["category"].value

            route_scores = self.routes.score_all(query_text, category_hint, FUSION_TOP_N_PER_ROUTE)
            route_weights = ROUTE_WEIGHTS_BUYING if state.track == "buying" else ROUTE_WEIGHTS_BROWSING
            fused = fusion.fuse(route_scores, route_weights)
            ranked = precision_pass.truncate(fused, PRECISION_PASS_TOP_K)
            scored = rank.rank_weighted(ranked, state, self.products)
            ranked = guarantee_pass.apply(scored, state, self.products)
            ranked = precision_pass.category_rescue(
                ranked, route_scores.get("category", [])[:CATEGORY_RESCUE_PICK_ZONE], state, self.products,
                CATEGORY_RESCUE_COUNT, CATEGORY_RESCUE_TOP_K,
            )

            domain_signal = constraint_engine.evaluate_domain(
                [asin for asin, _ in ranked], self.products, state
            )
            state.last_pool_size = domain_signal.size
            state.last_over_generality = domain_signal.over_generality
            state.last_collision = domain_signal.collision
            state.last_candidate_asins = [asin for asin, _ in ranked]

            decision = strategy.decide(state, ranked, domain_signal, self.products)
            state.strategy_history.append(decision.ask_attribute or "")

            ask_attribute = decision.ask_attribute if decision.ask_attribute in ALLOWED_ASK_ATTRIBUTES else None

            return {
                "message": decision.message or DEFAULT_MESSAGE,
                "ask_attribute": ask_attribute,
                "recommendations": [
                    {"parent_asin": asin, "score": float(score)} for asin, score in decision.recommendations
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
        except Exception:
            return {
                "message": DEFAULT_MESSAGE,
                "ask_attribute": None,
                "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

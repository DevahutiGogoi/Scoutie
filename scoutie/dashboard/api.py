"""Thin Flask wrapper around scoutie.agent.Agent for the live product UI. Out of scope for the evaluator, but useful for a public-facing demo.

Two response shapes, deliberately kept apart (see CLAUDE.md gotcha #8 and the UI build spec):

  * /api/session/<id>/message returns the real turn_response contract fields (message,
    ask_attribute, recommendations, usage) -- message is humanized for display, but the
    underlying decision (which attribute, which products) is exactly what agent.respond()
    returned, untouched.
  * /api/session/<id>/state is a SEPARATE, unscored debug/UI endpoint that reads the live
    SessionState directly for panel data (pool size, track, slot chips, status line) the
    evaluator contract has no room for. Nothing here ever gets added to the real turn_response
    dict, and nothing here feeds back into agent.respond()'s own decision-making.

Run with: python -m scoutie.dashboard.api
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from scoutie.agent import Agent
from scoutie.config.thresholds import (
    ASKABLE_ATTRIBUTES,
    OVER_GENERALITY_DOMAIN_SIZE,
    PRECISION_PASS_TOP_K,
)
from scoutie.dashboard import humanizer
from scoutie.dashboard.catalog_ingest import convert_csv_to_catalog
from scoutie.dashboard.live_adapter import adapt_message
from scoutie.strategy import BUCKET_FUNCTIONS
from scoutie.understanding.belief_revision import GENERIC_PUSHBACK
from scoutie.understanding.constraint_engine import match_text, stated_hard_slots

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_PATH = REPO_ROOT / "data" / "catalog.jsonl"
STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"

# The competition's own hard per-session turn cap (docs/competition_specification.md 4.3). Kept
# as an independent local constant, not imported from evaluator/ -- see CLAUDE.md's standing rule
# that scoutie/ never imports from evaluator/ or starter/ at runtime.
DASHBOARD_MAX_TURNS = 10

# Round 19 (public-facing demo redesign): each persona seeds a REAL user_profile -- not flavor
# text. purchase_frequency needs a leading digit >=1 to nudge track_fusion.py's Dempster-Shafer
# profile mass toward "buying" (see _profile_mass()'s own docstring); no digit leaves it neutral.
# preference_tags feeds rank.py's _profile_score() directly (fraction of tags found in a
# candidate's searchable text). Confirmed both mechanisms exist and read exactly these two keys
# before writing this -- see track_fusion.py/rank.py.
PERSONAS = [
    {
        "id": "minimalist",
        "name": "The Minimalist",
        "emoji": "\U0001F9E5",
        "tagline": "Buys rarely, wants durable basics that last.",
        "user_profile": {"purchase_frequency": "1 purchase in the last year", "preference_tags": ["classic", "durable", "neutral"]},
    },
    {
        "id": "trendsetter",
        "name": "The Trendsetter",
        "emoji": "✨",
        "tagline": "Shops often, always chasing what's new.",
        "user_profile": {"purchase_frequency": "6 purchases this year", "preference_tags": ["trendy", "statement", "colorful"]},
    },
    {
        "id": "practical",
        "name": "The Practical Parent",
        "emoji": "\U0001F45C",
        "tagline": "Shops for the whole family -- comfort and price matter most.",
        "user_profile": {"purchase_frequency": "4 purchases this year", "preference_tags": ["affordable", "comfortable", "practical"]},
    },
    {
        "id": "browsing",
        "name": "Just Browsing Today",
        "emoji": "\U0001F440",
        "tagline": "No fixed plan -- just seeing what's out there.",
        "user_profile": {"purchase_frequency": "no recent purchases", "preference_tags": []},
    },
]

# Turn-1 starter chips: deliberately a small, hand-picked set of category nouns known to retrieve
# well against this frozen catalog (Clothing/Shoes/Jewelry), not sampled live -- keeps turn 1's
# lowest-risk free-text step forgiving without needing catalog introspection at startup.
STARTER_CATEGORY_CHIPS = ["jacket", "shoes", "dress", "jewelry", "bag", "sweater"]

# Friendly labels + a representative dollar figure to actually send (belief_revision's budget
# matching is single-target-with-slack, not a true range -- see constraint_engine.BUDGET_DOMAIN_SLACK
# -- so each bucket sends its upper bound as an honest approximation of "up to about this much").
BUDGET_BUCKET_OPTIONS = [
    ("under_15", "Under $15", "under $15"),
    ("15_30", "$15 - $30", "under $30"),
    ("30_50", "$30 - $50", "under $50"),
    ("50_100", "$50 - $100", "under $100"),
    ("over_100", "Over $100", "over $100"),
]

# Attributes with a clean, real enumerable value set worth turning into quick-reply buttons.
# "feature" and "other" are deliberately excluded -- feature's bucket_fn returns the first few
# words of a random feature bullet (not a clean, tappable value), and "other" has no single
# attribute to enumerate against; both keep free text as the primary path.
# A sentinel the frontend sends verbatim for the "other" ask's "Nothing else, thanks" button --
# never something a real shopper would type, so it can't collide with genuine chat text.
NOTHING_ELSE_SENTINEL = "__scoutie_nothing_else__"

QUICK_REPLY_ATTRIBUTES = ("material", "color", "size", "style", "budget", "use_case")
MAX_QUICK_REPLY_VALUES = 4
# A legitimate material/color/size/style/use_case value is always short. Confirmed live: some
# catalog rows have their `details.Style` field populated with what looks like a duplicated
# product title rather than a real style descriptor -- _bucket_style()'s structured-field-first
# fallback (constraint_engine.py) trusts it as-is, which was harmless while nothing rendered
# these values directly, but becomes a visibly broken button the moment they're user-facing.
MAX_QUICK_REPLY_LABEL_LENGTH = 30
DOT_COUNT = 60

# Sourced from results/round17-category-usecase-fix_20260829_224447.json (round 17's
# category-hard-match fix; see PROGRESS.md's round-17 entry). Not fabricated -- these are the
# real, most recently verified full-evaluator numbers for the shipped agent, read once at import
# time so the outcome panel doesn't silently drift if that file is deleted between sessions.
# NOTE: this filename is a manual pointer, not auto-discovered -- every time a core-path change
# (constraint_engine.py, text_utils.py, etc.) gets a fresh evaluator verification, this must be
# updated to the new result file, or the dashboard silently shows a stale score again (as it did
# here for one round after the category fix landed).
_METRICS_FALLBACK = {"hit_rate_at_10": 0.945, "mrr": 0.594772, "mttc": 2.85}


def _load_aggregate_metrics() -> dict:
    import json

    results_path = REPO_ROOT / "results" / "round17-category-usecase-fix_20260829_224447.json"
    try:
        with results_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return {
            "hit_rate_at_10": data["hit_rate_at_10"],
            "mrr": data["mrr"],
            "mttc": data["mttc"],
        }
    except (OSError, KeyError, ValueError):
        return dict(_METRICS_FALLBACK)


AGGREGATE_METRICS = _load_aggregate_metrics()
METRIC_CARDS = [
    {
        "key": "hit_rate_at_10",
        "label": "Hit Rate @ 10",
        "value": AGGREGATE_METRICS["hit_rate_at_10"],
        "display": f"{AGGREGATE_METRICS['hit_rate_at_10'] * 100:.1f}%",
        "blurb": "How often Scoutie found the right item within the first 10 suggestions, across 200 test sessions.",
    },
    {
        "key": "mrr",
        "label": "MRR",
        "value": AGGREGATE_METRICS["mrr"],
        "display": f"{AGGREGATE_METRICS['mrr']:.2f}",
        "blurb": "How high up the list Scoutie ranked the right item when it found it -- 1.0 means first place, every time.",
    },
    {
        "key": "mttc",
        "label": "Avg. turns to convert",
        "value": AGGREGATE_METRICS["mttc"],
        "display": f"{AGGREGATE_METRICS['mttc']:.2f}",
        "blurb": "On average, how many messages it took to land on the right product, across 200 test sessions.",
    },
]


def create_app(catalog_path: str | Path | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    agent_holder: dict[str, Agent] = {"agent": Agent(str(catalog_path or DEFAULT_CATALOG_PATH))}
    turn_counters: dict[str, int] = {}
    converted_sessions: dict[str, dict] = {}

    def _agent() -> Agent:
        return agent_holder["agent"]

    def _product_card(parent_asin: str, score: float) -> dict:
        product = _agent().products.get(parent_asin, {})
        return {
            "parent_asin": parent_asin,
            "score": score,
            "title": product.get("title") or parent_asin,
            "price": product.get("price"),
            "store": product.get("store") or None,
            "average_rating": product.get("average_rating"),
        }

    def _status_line(state, *, converted: bool) -> str:
        if converted:
            return "Found your match."
        if state.turn_count <= 1:
            return "Reading what you're looking for"
        if state.last_pool_size > OVER_GENERALITY_DOMAIN_SIZE:
            return "Narrowing the options"
        return "Lining up your best matches"

    def _dot_fill(pool_size: int) -> int:
        import math

        ceiling = max(PRECISION_PASS_TOP_K, 1)
        fraction = min(1.0, math.log(pool_size + 1) / math.log(ceiling + 1))
        return round(fraction * DOT_COUNT)

    def _quick_replies_for(attribute: str | None, state) -> list[dict]:
        """Real values pulled from the CURRENT candidate pool (state.last_candidate_asins) --
        the exact same pool strategy.py's entropy-based asking already scores, via the exact
        same BUCKET_FUNCTIONS it uses -- not invented, not a fixed list. Empty when there's no
        clean attribute to enumerate (None/"other"/"feature") or the pool is empty, so the
        frontend falls back to free text."""
        if not attribute or attribute not in QUICK_REPLY_ATTRIBUTES:
            return []
        asins = state.last_candidate_asins
        if not asins:
            return []
        products = _agent().products

        if attribute == "budget":
            bucket_fn = BUCKET_FUNCTIONS["budget"]
            present_buckets = {bucket_fn(products.get(asin, {})) for asin in asins}
            present_buckets.discard(None)
            options = [
                {"label": label, "value": send_text}
                for bucket_id, label, send_text in BUDGET_BUCKET_OPTIONS
                if bucket_id in present_buckets
            ]
            return options[:MAX_QUICK_REPLY_VALUES]

        bucket_fn = BUCKET_FUNCTIONS[attribute]
        counts: dict[str, int] = {}
        for asin in asins:
            value = bucket_fn(products.get(asin, {}))
            if value and len(value) <= MAX_QUICK_REPLY_LABEL_LENGTH:
                counts[value] = counts.get(value, 0) + 1
        top_values = sorted(counts, key=lambda value: -counts[value])[:MAX_QUICK_REPLY_VALUES]
        return [{"label": value.title(), "value": value} for value in top_values]

    def _other_ask_suggestions(state) -> list[dict]:
        """The broad "other" ask (see ASK_OTHER_MAX_COUNT in config/thresholds.py) covers most of
        a fresh session's first few turns -- exactly where a public visitor most needs a safe
        click path, and exactly the one ask_attribute QUICK_REPLY_ATTRIBUTES deliberately
        excludes (there's no single attribute to enumerate against). This merges the single top
        real value from each of a few attributes into one suggestion row instead. Safe by the
        same mechanism gotcha #9 documents: customer_reply()'s match filter (and this dashboard's
        classify_constraint()-based reclassification) resolves each "other"-disclosed value
        independently by its own content, not by which button it happened to sit under."""
        suggestions = []
        for attribute in ("material", "color", "budget", "use_case"):
            options = _quick_replies_for(attribute, state)
            if options:
                suggestions.append(options[0])
        return suggestions

    def _prior_ask_attribute(state) -> str | None:
        if len(state.strategy_history) < 2:
            return None
        value = state.strategy_history[-2]
        return value or None

    def _just_captured_value(state, *, just_overrode: bool, just_declined: bool, just_exhausted: bool) -> str | None:
        if just_overrode or just_declined or just_exhausted:
            return None
        captured = [
            slot.value
            for attribute, slot in state.slots.items()
            if attribute != "category"
            and slot.status == "stated"
            and slot.value
            and slot.last_updated_turn == state.turn_count
        ]
        if not captured:
            return None
        value = captured[0]
        return value if len(value) <= 40 else value[:37] + "..."

    def _unknown_attribute_phrases(state) -> tuple[str, ...]:
        return tuple(
            humanizer.ATTRIBUTE_PHRASES.get(attribute, attribute.replace("_", " "))
            for attribute in ASKABLE_ATTRIBUTES
            if state.slots[attribute].status == "unknown"
        )

    def _build_turn_event(state, ask_attribute: str | None, *, converted: bool) -> humanizer.TurnEvent:
        prior_ask = _prior_ask_attribute(state)
        just_overrode = bool(state.override_events) and state.override_events[-1]["turn"] == state.turn_count
        just_declined = False
        just_exhausted = False
        if prior_ask and prior_ask != "other" and prior_ask in state.slots:
            slot = state.slots[prior_ask]
            just_declined = slot.status == "no_preference" and slot.last_updated_turn == state.turn_count
            just_exhausted = slot.status == "exhausted" and slot.last_updated_turn == state.turn_count
        return humanizer.TurnEvent(
            turn_count=state.turn_count,
            ask_attribute=ask_attribute,
            over_generality=state.last_over_generality,
            collision=state.last_collision,
            just_overrode=just_overrode,
            just_declined=just_declined,
            just_exhausted=just_exhausted,
            had_prior_ask=prior_ask is not None,
            converted=converted,
            just_captured_value=_just_captured_value(
                state, just_overrode=just_overrode, just_declined=just_declined, just_exhausted=just_exhausted
            ),
            unknown_attribute_phrases=_unknown_attribute_phrases(state),
        )

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/static/<path:filename>")
    def static_files(filename: str):
        return send_from_directory(STATIC_DIR, filename)

    @app.post("/api/session")
    def start_session():
        body = request.get_json(silent=True) or {}
        session_id = str(uuid.uuid4())
        _agent().reset(session_id, body.get("user_profile") or {})
        turn_counters[session_id] = 0
        return jsonify({"session_id": session_id, "max_turns": DASHBOARD_MAX_TURNS})

    @app.post("/api/session/<session_id>/message")
    def post_message(session_id: str):
        if session_id not in _agent().sessions:
            return jsonify({"error": "unknown session_id"}), 404
        body = request.get_json(silent=True) or {}
        raw_message = (body.get("message") or "").strip()
        if not raw_message:
            return jsonify({"error": "message is required"}), 400

        turn = turn_counters.get(session_id, 0) + 1
        if turn > DASHBOARD_MAX_TURNS:
            return jsonify({"error": "session has reached the turn limit"}), 409
        turn_counters[session_id] = turn

        state = _agent().sessions[session_id]
        prior_ask = state.strategy_history[-1] if state.strategy_history else None
        if raw_message == NOTHING_ELSE_SENTINEL:
            # The "other" ask's own quick-reply row (see _other_ask_suggestions) offers this as a
            # button. Bypasses live_adapter/classify_constraint entirely and goes straight to
            # belief_revision's own designed-in no-op string -- there's no clean decline mechanism
            # for an "other" ask otherwise (only a SPECIFIC asked attribute has a scripted
            # boundary-decline phrase), so routing plain "no preference" text through the normal
            # disclosure path would misfire.
            adapted = GENERIC_PUSHBACK
        else:
            adapted = adapt_message(raw_message, turn, prior_ask or None)

        response = _agent().respond(session_id, adapted, turn, 10)
        state = _agent().sessions[session_id]  # respond() may have re-created it on first call

        event = _build_turn_event(state, response.get("ask_attribute"), converted=False)
        display_message = humanizer.humanize(event)

        recommendations = [
            _product_card(item["parent_asin"], item["score"]) for item in response.get("recommendations", [])[:6]
        ]

        return jsonify(
            {
                "message": display_message,
                "ask_attribute": response.get("ask_attribute"),
                "recommendations": recommendations,
                "turn_count": turn,
                "done": turn >= DASHBOARD_MAX_TURNS,
            }
        )

    @app.post("/api/session/<session_id>/select")
    def select_recommendation(session_id: str):
        if session_id not in _agent().sessions:
            return jsonify({"error": "unknown session_id"}), 404
        body = request.get_json(silent=True) or {}
        parent_asin = body.get("parent_asin")
        if not parent_asin:
            return jsonify({"error": "parent_asin is required"}), 400

        state = _agent().sessions[session_id]
        converted_sessions[session_id] = {"parent_asin": parent_asin, "turn_count": state.turn_count}
        event = _build_turn_event(state, None, converted=True)
        close_message = humanizer.humanize(event)

        # Genuinely checkable per-session fact, unlike hit-rate/MRR (which need a labeled ground
        # truth this live session doesn't have -- see the outcome panel's own note). Reuses the
        # exact same match_text()/stated_hard_slots() guarantee_pass.py itself boosts/demotes on,
        # so "satisfies" here means the same thing it means everywhere else in the pipeline.
        product = _agent().products.get(parent_asin, {})
        stated = stated_hard_slots(state)
        violated_attributes = [
            attribute
            for attribute, value in stated
            if match_text(attribute, value, product, state.track) == "violate"
        ]

        return jsonify(
            {
                "message": close_message,
                "product": _product_card(parent_asin, 0.0),
                "turn_count": state.turn_count,
                "satisfies_all_constraints": not violated_attributes,
                "violated_attributes": violated_attributes,
                "metrics": METRIC_CARDS,
            }
        )

    @app.get("/api/session/<session_id>/state")
    def session_state(session_id: str):
        if session_id not in _agent().sessions:
            return jsonify({"error": "unknown session_id"}), 404
        state = _agent().sessions[session_id]
        converted = session_id in converted_sessions

        chips = [
            {"attribute": attribute, "value": slot.value}
            for attribute, slot in state.slots.items()
            if slot.status in ("stated", "overridden") and slot.value
        ]

        current_ask = state.strategy_history[-1] if state.strategy_history else None
        current_ask = current_ask or None
        # Computed for every quick-reply-eligible attribute, not just the one currently being
        # asked -- the frontend uses this both to render buttons for the live question AND to
        # power click-to-edit on an already-answered chip (revise any attribute, any time).
        quick_reply_options = {
            attribute: _quick_replies_for(attribute, state) for attribute in QUICK_REPLY_ATTRIBUTES
        }
        quick_reply_options["other"] = _other_ask_suggestions(state)

        return jsonify(
            {
                "status_line": _status_line(state, converted=converted),
                "mode": "Ready to buy" if state.track == "buying" else "Just browsing",
                "turn_count": state.turn_count,
                "max_turns": DASHBOARD_MAX_TURNS,
                "options_left": state.last_pool_size,
                "dot_fill": _dot_fill(state.last_pool_size),
                "dot_count": DOT_COUNT,
                "current_ask_attribute": current_ask,
                "quick_reply_options": quick_reply_options,
                "no_preference_label": "No preference",
                "slots": chips,
                "converted": converted,
            }
        )

    @app.get("/api/metrics")
    def metrics():
        return jsonify({"metrics": METRIC_CARDS})

    @app.get("/api/personas")
    def personas():
        return jsonify({"personas": PERSONAS, "starter_categories": STARTER_CATEGORY_CHIPS})

    @app.post("/api/session/<session_id>/revise")
    def revise_slot(session_id: str):
        """Click-to-edit for an already-answered chip. Writes state.slots[attribute] directly
        instead of routing through belief_revision.py's text parsing at all -- tried the obvious
        approach first (belief_revision's real OVERRIDE_RE scripted format) and found a real bug
        live: OVERRIDE_RE's handler calls _clear_stale_pre_override_slots(), which clears any
        OTHER stated slot whose value happens to be a verbatim substring of turn 1's own stored
        message. That heuristic is correctly tuned for the evaluator's genuine Intent Override
        scenarios (see belief_revision.py's own docstring), but a click-to-edit on ONE specific
        chip has nothing to do with that mechanism -- it silently wiped budget when revising
        color, just because "under $150" happened to still be sitting in turn 1's text. This
        action already knows exactly which attribute and value with full confidence (the chip's
        own attribute, a real value straight from _quick_replies_for()'s bucket-function output)
        -- there's no ambiguity for belief_revision's parsing to resolve, so it's safer to just
        write the slot the same way belief_revision's own _assign_slot()/_record_override() would,
        then let agent.respond() re-run retrieval/ranking/strategy from the corrected state.
        GENERIC_PUSHBACK is belief_revision's own designed-in no-op string, so passing it through
        respond() refreshes recommendations without belief_revision reinterpreting or clobbering
        the direct slot write above."""
        if session_id not in _agent().sessions:
            return jsonify({"error": "unknown session_id"}), 404
        body = request.get_json(silent=True) or {}
        attribute = body.get("attribute")
        value = (body.get("value") or "").strip()
        if not attribute or not value:
            return jsonify({"error": "attribute and value are required"}), 400
        state = _agent().sessions[session_id]
        if attribute not in state.slots:
            return jsonify({"error": "unknown attribute"}), 400

        turn = turn_counters.get(session_id, 0) + 1
        if turn > DASHBOARD_MAX_TURNS:
            return jsonify({"error": "session has reached the turn limit"}), 409
        turn_counters[session_id] = turn

        slot = state.slots[attribute]
        if slot.status in ("stated", "overridden") and slot.value and slot.value != value:
            state.override_events.append({"turn": turn, "attribute": attribute, "old": slot.value, "new": value})
            slot.status = "overridden"
        else:
            slot.status = "stated"
        slot.value = value
        slot.weight = 1.0
        slot.last_updated_turn = turn

        response = _agent().respond(session_id, GENERIC_PUSHBACK, turn, 10)
        state = _agent().sessions[session_id]

        event = _build_turn_event(state, response.get("ask_attribute"), converted=False)
        display_message = humanizer.humanize(event)
        recommendations = [
            _product_card(item["parent_asin"], item["score"]) for item in response.get("recommendations", [])[:6]
        ]
        return jsonify(
            {
                "message": display_message,
                "ask_attribute": response.get("ask_attribute"),
                "recommendations": recommendations,
                "turn_count": turn,
                "done": turn >= DASHBOARD_MAX_TURNS,
            }
        )

    @app.post("/api/catalog/upload")
    def upload_catalog():
        # This endpoint rebuilds the ONE shared Agent instance for every visitor -- fine on a
        # private LAN demo, a real risk once the server is reachable over a public tunnel (any
        # visitor could otherwise replace the catalog for everyone else mid-demo). Gated behind
        # an admin token that must be explicitly set; unset (the default) disables the endpoint
        # entirely rather than leaving it open.
        admin_token = os.environ.get("SCOUTIE_ADMIN_TOKEN")
        if not admin_token or request.headers.get("X-Admin-Token") != admin_token:
            return jsonify({"error": "not authorized"}), 403
        if "file" not in request.files:
            return jsonify({"error": "no file uploaded"}), 400
        upload = request.files["file"]
        if not upload.filename:
            return jsonify({"error": "empty filename"}), 400

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = UPLOAD_DIR / upload.filename
        upload.save(csv_path)
        catalog_path = UPLOAD_DIR / (Path(upload.filename).stem + ".catalog.jsonl")

        try:
            product_count = convert_csv_to_catalog(csv_path, catalog_path)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        if product_count == 0:
            return jsonify({"error": "no valid product rows found in that file"}), 400

        # Rebuilding the three retrieval routes over a new catalog is a real, synchronous cost
        # (BM25 index + two TF-IDF indexes over every row) -- acceptable for a one-time catalog
        # swap, not something to hide behind a fast-looking response.
        agent_holder["agent"] = Agent(str(catalog_path))
        turn_counters.clear()
        converted_sessions.clear()

        return jsonify({"product_count": product_count, "catalog_path": str(catalog_path)})

    return app


def main() -> None:
    app = create_app(os.environ.get("SCOUTIE_CATALOG_PATH"))
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5050")), threaded=True)


if __name__ == "__main__":
    main()

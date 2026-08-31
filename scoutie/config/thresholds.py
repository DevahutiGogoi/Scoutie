"""Every tunable number for Scoutie lives here. Nothing tunable is hardcoded elsewhere."""

from __future__ import annotations

# All nine attributes the evaluator's customer simulator can disclose against.
SLOT_ATTRIBUTES = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case",
)

# Attributes worth asking about. Excludes "category" (always disclosed free in turn 1) and
# "brand"/"other" (evaluator's classify_constraint() can never produce them from a disclosure,
# so asking about them via ask_attribute's normal slot-status path would only ever waste a turn --
# "other" is asked separately, see ASK_OTHER_MAX_COUNT below). Ordered so "feature" -- the
# fuzziest, highest-recall bucket -- is the fallback, not the first choice.
ASKABLE_ATTRIBUTES = ("material", "color", "size", "style", "budget", "use_case", "feature")

# --- Constraint engine (domain-size signals) ---
OVER_GENERALITY_DOMAIN_SIZE = 20   # domain > this -> over_generality flag
COLLISION_DOMAIN_SIZE = 3          # domain < this with a genuine contradiction -> collision flag

# --- Orchestration ---
DEFAULT_MESSAGE = "Here are my best matches so far."
ASK_TURN_CAP = 7                   # ask while turn_count < this; guess-only afterward
RECOMMENDATIONS_RETURN_COUNT = 15  # evaluator only scores the first 10 unique valid ids
MIN_POOL_FOR_ASK_HEURISTIC = 5     # below this, skip the entropy heuristic, ask "feature"

# customer_reply()'s match filter (evaluator/local_evaluator.py) collapses to "any not-yet-
# disclosed constraint, any type" when the asked attribute is "other" -- asking it reveals
# substantially more per turn than asking a specific attribute. Capped (not asked every turn) to
# limit a real, measured side effect: for intent_override sessions, asking "other" can disclose
# the override's own soon-to-be-repudiated `old_value` into a slot before the override fires.
ASK_OTHER_MAX_COUNT = 3

# --- Retrieval routes / fusion ---
FUSION_TOP_N_PER_ROUTE = 300
# Buying: precision-first (keyword-heavy), matching a high-precision filter track.
ROUTE_WEIGHTS_BUYING = {"keyword": 0.95, "category": 0.05, "vector": 0.0}
# Browsing: shifts weight toward VectorRoute (TF-IDF cosine similarity -- see routes.py) at the
# expense of keyword, for a more diverse retrieval track on open-ended queries.
ROUTE_WEIGHTS_BROWSING = {"keyword": 0.75, "category": 0.05, "vector": 0.20}
RRF_K = 60  # standard Reciprocal Rank Fusion constant (Cormack et al. 2009)
PRECISION_PASS_TOP_K = 60

# --- Ranking ---
RANK_WEIGHTS = {"fused": 0.65, "soft_slot": 0.15, "profile": 0.05, "popularity": 0.15}
POPULARITY_RATING_FLOOR = 3.0      # products rated below this get no popularity bonus
# Computed once over the full frozen 50,000-product catalog; every product had a rating
# (50000/50000), so no missing-data handling was needed for this constant itself.
CATALOG_MEAN_RATING = 4.087104
# "Virtual review count" the catalog mean is worth in the Bayesian shrinkage blend -- a
# thinly-reviewed product's rating is pulled toward CATALOG_MEAN_RATING; a well-established one
# is barely moved. shrunk_rating = (rating_number * rating + POPULARITY_BAYESIAN_PRIOR_STRENGTH *
# CATALOG_MEAN_RATING) / (rating_number + POPULARITY_BAYESIAN_PRIOR_STRENGTH).
POPULARITY_BAYESIAN_PRIOR_STRENGTH = 100.0

# --- Category rescue ---
# Reserves a slot in the actually-scored top-10 for CategoryRoute's own best pick if it isn't
# already there -- recovers targets the keyword-dominant fused ranking buries outside the window.
CATEGORY_RESCUE_COUNT = 1     # how many of the final top-K slots may be reserved for a rescue
CATEGORY_RESCUE_PICK_ZONE = 50  # how deep into CategoryRoute's own ranking to look for a rescue
CATEGORY_RESCUE_TOP_K = 10    # the actually-scored window (evaluator.local_evaluator.TOP_K), not
                               # RECOMMENDATIONS_RETURN_COUNT's wider 15

# --- Category hard-matching ---
# Round 17 (dashboard live-testing finding, Severity 1): category was structurally excluded from
# stated_hard_slots() everywhere (guarantee_pass, rank.py's soft_slot score,
# precision_pass.category_rescue, constraint_engine.evaluate_domain all read the same function) --
# meaning nothing downstream of retrieval ever checked that a candidate was even the right kind of
# product. Confirmed live: a session stating "jacket" ended with a t-shirt as the top-ranked,
# user-confirmed pick, because material/budget/use_case matching alone let it through as
# "non-violating" (its category was simply never examined). Flagged, not unconditional, because
# this changes constraint_engine.py's behavior on the SCORED evaluator path too (every consumer of
# stated_hard_slots() is shared) -- verify against the full evaluator before trusting the default,
# per this project's own standing discipline, and keep this reachable as an instant revert if it
# ever regresses on the private set.
CATEGORY_HARD_MATCH_ENABLED = True

# --- Slot decay ---
# Applies ONLY to rank.py's soft_slot ranking component's per-slot contribution -- never to
# constraint_engine.py's hard-filter or guarantee_pass.py's boost, both driven purely by
# Slot.status, never Slot.weight. A stale, unreaffirmed slot still counts as a hard constraint at
# full strength; it just contributes less to the ranking tie-breaker among already-hard-
# constraint-satisfying candidates.
SLOT_DECAY_RATE = 0.85    # multiplicative decay per turn since last_updated_turn, e.g. 0.85**3 =~ 0.61
SLOT_DECAY_FLOOR = 0.3    # a stale slot never drops below 30% of its original weight

# --- Constraint matching ---
BUDGET_EXACT_EPSILON = 0.01        # dollars; near-exact float match tolerance
BUDGET_DOMAIN_SLACK = 0.20         # 20% slack band before a budget constraint is "violate"
FUZZY_TOKEN_OVERLAP_THRESHOLD = 0.6

# --- Track fusion (buying vs. browsing) ---
TRANSACTIONAL_KEYWORDS = ("buy", "need", "want", "purchase", "require", "must have", "looking to buy")
EXPLORATORY_KEYWORDS = ("exploring", "browsing", "just looking", "not sure", "maybe", "curious", "still exploring")
BUYING_RATCHET = True              # once track == "buying", never revert to browsing

# Supplementary keyword/marker tiers below (TRANSACTIONAL/EXPLORATORY/HEDGE/CERTAINTY) exist for
# private-set phrasing beyond what the public customer simulator's own templates use -- kept
# separate from the AUDITED tiers so it's clear which vocabulary is directly evidence-grounded.
TRANSACTIONAL_KEYWORDS_SUPPLEMENTARY = (
    "buy", "purchase", "need a", "need one", "want to get", "get me", "order", "add to cart",
    "checkout", "ready to buy", "looking to buy", "want to buy", "gotta get", "must have",
    "shopping for a", "replace my", "upgrade my", "buying a", "i'll take", "sold on",
)
EXPLORATORY_KEYWORDS_SUPPLEMENTARY = (
    "looking for ideas", "just browsing", "just looking", "what do you have", "show me options",
    "what's available", "exploring", "curious about", "thinking about", "not sure yet",
    "window shopping", "seeing what's out there", "any recommendations", "what would you suggest",
    "open to suggestions", "just curious", "checking out", "what do you recommend",
)

# Hedge/certainty ("conviction") markers. AUDITED tiers are built directly from the customer
# simulator's own generated text and are empty by design on this dataset -- see
# track_fusion.py's _strip_templated_value() for why a naive scan would otherwise false-positive
# on quoted catalog text (e.g. "100% Cotton" material composition, not customer certainty).
HEDGE_MARKERS_AUDITED: tuple[str, ...] = ()
CERTAINTY_MARKERS_AUDITED: tuple[str, ...] = ()
HEDGE_MARKERS_SUPPLEMENTARY = (
    "might", "may", "could",
    "sort of", "kind of", "somewhat", "roughly", "around", "about", "more or less", "give or take",
    "fairly", "relatively", "a bit", "a little",
    "i think", "i guess", "i believe", "i'd say", "i suppose", "i assume", "i imagine",
    "i feel like", "my guess is", "if i had to guess",
    "apparently", "supposedly", "from what i understand", "something like", "or something",
    "or whatever", "along those lines", "in that ballpark",
    "probably", "possibly", "likely", "perhaps", "maybe", "presumably", "conceivably",
    "not sure", "not totally sure", "undecided", "still deciding", "haven't decided",
    "open to", "flexible on",
)
CERTAINTY_MARKERS_SUPPLEMENTARY = (
    "must", "has to", "have to", "need to", "needs to be", "required to be",
    "absolutely", "definitely", "certainly", "surely", "undoubtedly", "without question",
    "without a doubt", "100%", "completely", "totally", "for sure", "for certain",
    "exactly", "precisely", "specifically", "strictly", "only", "exclusively",
    "no more than", "no less than", "at least", "at most", "minimum", "maximum", "cap of",
    "under", "up to",
    "has to be", "will only", "won't accept", "refuse to", "insist on", "must-have",
    "deal-breaker", "non-negotiable", "always",
)

# Dempster-Shafer combination strengths: how much mass one piece of evidence assigns to its
# favored hypothesis vs. leaving as uncertain (Theta).
MESSAGE_MASS_BUYING_STRENGTH = 0.75
MESSAGE_MASS_BROWSING_STRENGTH = 0.65
MESSAGE_MASS_CONFLICT_SPLIT = 0.3   # symmetric mass to each side when one message's clean text
                                     # trips both transactional and exploratory cues at once
CONVICTION_MASS_STRENGTH = 0.4
# Deliberately mild: purchase_frequency is a free-text profile field with no guaranteed numeric
# signal, so this mass is forward-looking (private-set generality), not a strong discriminator.
PROFILE_MASS_STRENGTH = 0.15
TRACK_FUSION_BUYING_BELIEF_THRESHOLD = 0.65    # pignistic P(buying) needed to ratchet to "buying"
TRACK_FUSION_BROWSING_BELIEF_THRESHOLD = 0.55  # pignistic P(browsing) for provisional (non-ratcheted) "browsing"

# --- Distillation ---
DISTILLATION_CONTEXT_TURNS = 3     # last N raw turn messages kept as retrieval query context

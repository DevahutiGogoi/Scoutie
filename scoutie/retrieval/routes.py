"""Three independent retrieval scorers, built once at Agent construction and reused every turn."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from pathlib import Path

import numpy as np

from scoutie.text_utils import _terms, _text

BM25_COLUMN_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
VECTOR_FIELDS = ("title", "features", "description", "categories", "details", "store")


def load_catalog_rows(catalog_path: str | Path) -> tuple[list[str], list[dict]]:
    """Single read of the catalog, giving every route the same doc_id order to key off."""
    doc_ids: list[str] = []
    rows: list[dict] = []
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            doc_ids.append(str(product["parent_asin"]))
            rows.append(product)
    return doc_ids, rows


class KeywordRoute:
    """Reuses the starter agent's exact FTS5 BM25 setup, stdlib-only.

    check_same_thread=False + an explicit lock around every post-init query: sqlite3's default
    same-thread restriction is fine for the single-process evaluator run, but Agent is built once
    at process startup and then reused across request-handling threads by the dashboard API's
    threaded Flask server (scoutie/dashboard/api.py) -- every score() call from a non-owning
    thread would otherwise raise sqlite3.ProgrammingError, which agent.py's blanket
    `except Exception` in respond() silently swallows into an empty-recommendations fallback (no
    error surfaced, no crash -- just a session that quietly never gets any keyword-route results).
    The lock matters as much as the flag: SQLite itself still needs single-connection access
    serialized across threads even once check_same_thread stops blocking it outright. The
    write phase (index population) below happens entirely in __init__, before any other thread
    can reach this object, so no lock is needed there -- only score()'s reads need it.
    """

    def __init__(self, doc_ids: list[str], rows: list[dict]) -> None:
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._lock = threading.Lock()
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        for doc_id, product in zip(doc_ids, rows):
            batch.append(
                (
                    doc_id,
                    _text(product.get("title")),
                    _text(product.get("categories")),
                    _text(product.get("features")),
                    _text(product.get("details")),
                    _text(product.get("store")),
                    _text(product.get("description")),
                )
            )
            if len(batch) >= 1000:
                cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def score(self, query_text: str, top_n: int) -> list[tuple[str, float]]:
        unique_terms = list(dict.fromkeys(_terms(query_text)))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            return []
        weights_sql = ", ".join(str(weight) for weight in BM25_COLUMN_WEIGHTS)
        with self._lock:
            rows = self.connection.execute(
                f"SELECT parent_asin, bm25(products, {weights_sql}) FROM products "
                "WHERE products MATCH ? ORDER BY bm25(products, " + weights_sql + ") LIMIT ?",
                (expression, top_n),
            ).fetchall()
        # sqlite's bm25() is more-negative-is-better; negate so every route is higher-is-better.
        return [(str(parent_asin), -float(raw_score)) for parent_asin, raw_score in rows]


def _top_n_from_scores(scores: np.ndarray, doc_ids: list[str], top_n: int) -> list[tuple[str, float]]:
    n_docs = scores.shape[0]
    if top_n >= n_docs:
        top_indices = np.argsort(-scores)
    else:
        candidates = np.argpartition(-scores, top_n)[:top_n]
        top_indices = candidates[np.argsort(-scores[candidates])]
    results: list[tuple[str, float]] = []
    for index in top_indices:
        value = float(scores[index])
        if value <= 0.0:
            break  # scores are non-negative; once we hit 0 the rest (sorted desc) are also 0
        results.append((doc_ids[index], value))
    return results


class CategoryRoute:
    """IDF-weighted token overlap between (category hint + query terms) and each doc's category tokens."""

    def __init__(self, doc_ids: list[str], rows: list[dict]) -> None:
        self.doc_ids = doc_ids
        n_docs = len(doc_ids)
        vocab: dict[str, int] = {}
        df_counts: list[int] = []
        doc_token_ids: list[set[int]] = []
        for product in rows:
            tokens = set(_terms(_text(product.get("categories"))))
            token_ids: set[int] = set()
            for token in tokens:
                term_id = vocab.get(token)
                if term_id is None:
                    term_id = len(vocab)
                    vocab[token] = term_id
                    df_counts.append(0)
                token_ids.add(term_id)
            for term_id in token_ids:
                df_counts[term_id] += 1
            doc_token_ids.append(token_ids)

        idf = np.empty(len(vocab), dtype=np.float32)
        for term_id, doc_freq in enumerate(df_counts):
            idf[term_id] = math.log((n_docs + 1) / (doc_freq + 1)) + 1.0

        postings: list[list[int]] = [[] for _ in range(len(vocab))]
        for doc_index, token_ids in enumerate(doc_token_ids):
            for term_id in token_ids:
                postings[term_id].append(doc_index)

        self.vocab = vocab
        self.idf = idf
        self.postings = [np.array(p, dtype=np.int32) for p in postings]
        self.n_docs = n_docs

    def score(self, category_hint: str | None, query_terms: list[str], top_n: int) -> list[tuple[str, float]]:
        tokens = set(_terms(category_hint or "")) | set(query_terms or [])
        if not tokens:
            return []
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for token in tokens:
            term_id = self.vocab.get(token)
            if term_id is None:
                continue
            postings = self.postings[term_id]
            if postings.size:
                scores[postings] += self.idf[term_id]
        return _top_n_from_scores(scores, self.doc_ids, top_n)


class VectorRoute:
    """Hand-rolled TF-IDF cosine similarity via a sparse inverted index (no dense matrix, no sklearn).

    Built once at construction (catalog is frozen for the whole run) and reused every respond().
    """

    def __init__(self, doc_ids: list[str], rows: list[dict]) -> None:
        self.doc_ids = doc_ids
        n_docs = len(doc_ids)
        vocab: dict[str, int] = {}
        df_counts: list[int] = []
        doc_term_counts: list[dict[int, int]] = []

        for product in rows:
            text = " ".join(_text(product.get(field)) for field in VECTOR_FIELDS)
            counts: dict[int, int] = {}
            for term in _terms(text):
                term_id = vocab.get(term)
                if term_id is None:
                    term_id = len(vocab)
                    vocab[term] = term_id
                    df_counts.append(0)
                counts[term_id] = counts.get(term_id, 0) + 1
            for term_id in counts:
                df_counts[term_id] += 1
            doc_term_counts.append(counts)

        idf = np.empty(len(vocab), dtype=np.float32)
        for term_id, doc_freq in enumerate(df_counts):
            idf[term_id] = math.log((n_docs + 1) / (doc_freq + 1)) + 1.0

        postings_doc_ids: list[list[int]] = [[] for _ in range(len(vocab))]
        postings_weights: list[list[float]] = [[] for _ in range(len(vocab))]
        for doc_index, counts in enumerate(doc_term_counts):
            weights = {term_id: (1.0 + math.log(tf)) * idf[term_id] for term_id, tf in counts.items()}
            norm = math.sqrt(sum(weight * weight for weight in weights.values())) or 1.0
            for term_id, weight in weights.items():
                postings_doc_ids[term_id].append(doc_index)
                postings_weights[term_id].append(weight / norm)

        self.vocab = vocab
        self.idf = idf
        self.postings_doc_ids = [np.array(p, dtype=np.int32) for p in postings_doc_ids]
        self.postings_weights = [np.array(w, dtype=np.float32) for w in postings_weights]
        self.n_docs = n_docs

    def score(self, query_text: str, top_n: int) -> list[tuple[str, float]]:
        query_counts: dict[int, int] = {}
        for term in _terms(query_text):
            term_id = self.vocab.get(term)
            if term_id is None:
                continue
            query_counts[term_id] = query_counts.get(term_id, 0) + 1
        if not query_counts:
            return []
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for term_id, tf in query_counts.items():
            query_weight = (1.0 + math.log(tf)) * self.idf[term_id]
            doc_ids_arr = self.postings_doc_ids[term_id]
            if doc_ids_arr.size:
                scores[doc_ids_arr] += self.postings_weights[term_id] * query_weight
        return _top_n_from_scores(scores, self.doc_ids, top_n)


class Routes:
    """Built exactly once in Agent.__init__, held for the process lifetime, reused every turn."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.doc_ids, self.rows = load_catalog_rows(catalog_path)
        self.keyword = KeywordRoute(self.doc_ids, self.rows)
        self.category = CategoryRoute(self.doc_ids, self.rows)
        self.vector = VectorRoute(self.doc_ids, self.rows)

    def score_all(self, query_text: str, category_hint: str | None, top_n_per_route: int) -> dict[str, list[tuple[str, float]]]:
        query_terms = _terms(query_text)
        return {
            "keyword": self.keyword.score(query_text, top_n_per_route),
            "category": self.category.score(category_hint, query_terms, top_n_per_route),
            "vector": self.vector.score(query_text, top_n_per_route),
        }

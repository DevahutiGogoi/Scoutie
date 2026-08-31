"""CSV -> the JSONL schema Agent(catalog_path) expects (scoutie/retrieval/routes.py's
load_catalog_rows()), so a real seller's export can replace the frozen competition catalog.

Schema derived directly from data/catalog.jsonl's own rows (read at build time, not guessed):
parent_asin, title, features (list[str]), description (list[str]), price (float|null),
categories (list[str]), details (dict), average_rating (float), rating_number (int), store (str).

Only parent_asin and title are required from the source CSV; everything else defaults to an
empty/neutral value so a minimal seller export still produces valid rows. Column names are
matched case-insensitively with a small set of common synonyms (a Shopify export uses "Handle"/
"Title"/"Variant Price"/"Vendor"/"Product Type", not this competition kit's Amazon-derived names).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

COLUMN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "parent_asin": ("parent_asin", "asin", "sku", "id", "handle", "product id"),
    "title": ("title", "name", "product title", "product name"),
    "price": ("price", "variant price", "cost"),
    "features": ("features", "bullet points", "highlights"),
    "description": ("description", "body", "body (html)", "product description"),
    "categories": ("categories", "category", "product type", "collection", "type"),
    "store": ("store", "brand", "vendor", "manufacturer", "seller"),
    "average_rating": ("average_rating", "rating", "avg rating"),
    "rating_number": ("rating_number", "review count", "num reviews", "rating count"),
}


def _find_column(fieldnames: list[str], synonyms: tuple[str, ...]) -> str | None:
    lowered = {name.strip().lower(): name for name in fieldnames}
    for synonym in synonyms:
        if synonym in lowered:
            return lowered[synonym]
    return None


def _split_list_field(raw: str | None) -> list[str]:
    if not raw or not raw.strip():
        return []
    for separator in ("|", ";", "\n"):
        if separator in raw:
            return [part.strip() for part in raw.split(separator) if part.strip()]
    return [raw.strip()]


def _parse_price(raw: str | None) -> float | None:
    if not raw or not raw.strip():
        return None
    cleaned = raw.strip().lstrip("$").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def convert_csv_to_catalog(csv_path: str | Path, out_path: str | Path) -> int:
    """Returns the number of product rows written."""
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        columns = {key: _find_column(fieldnames, synonyms) for key, synonyms in COLUMN_SYNONYMS.items()}
        if columns["title"] is None:
            raise ValueError("CSV has no recognizable title/name column -- cannot build a catalog from it.")

        written = 0
        with Path(out_path).open("w", encoding="utf-8") as out_handle:
            for index, row in enumerate(reader):
                title = (row.get(columns["title"]) or "").strip()
                if not title:
                    continue
                parent_asin = (row.get(columns["parent_asin"]) or "").strip() if columns["parent_asin"] else ""
                product = {
                    "parent_asin": parent_asin or f"SELLER-{index:06d}",
                    "title": title,
                    "features": _split_list_field(row.get(columns["features"]) if columns["features"] else None),
                    "description": _split_list_field(row.get(columns["description"]) if columns["description"] else None),
                    "price": _parse_price(row.get(columns["price"]) if columns["price"] else None),
                    "categories": _split_list_field(row.get(columns["categories"]) if columns["categories"] else None),
                    "details": {},
                    "average_rating": float(row[columns["average_rating"]]) if columns["average_rating"] and row.get(columns["average_rating"]) else 0.0,
                    "rating_number": int(float(row[columns["rating_number"]])) if columns["rating_number"] and row.get(columns["rating_number"]) else 0,
                    "store": (row.get(columns["store"]) or "").strip() if columns["store"] else "",
                }
                out_handle.write(json.dumps(product) + "\n")
                written += 1
        return written

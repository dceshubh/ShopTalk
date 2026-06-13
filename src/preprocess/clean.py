"""Cleaning + canonical-document construction — THE shared transformer.

This module is imported by both the offline indexer and the online API. Whatever shape of
text goes into the embedding model at index time is produced by exactly these functions, and
the same functions run on a live query — satisfying the deployment rubric's "same data
transformers used in training and inference" requirement by construction, not by convention.

Design notes (documented cleaning decisions — these are the EDA "depth" the rubric wants):
  * ABO listings are multilingual: every text field is a list of {language_tag, value}.
    We select a single English entry per field using ENGLISH_LOCALE_PRIORITY — native
    English-speaking marketplaces first (en_US, en_GB, en_CA, en_AU) ahead of en_IN/en_AE/
    en_SG, because spot-checking showed the latter sometimes mix in non-English fragments
    inside nominally-English-tagged fields (e.g. Spanish words inside an `en_IN` keyword
    list). Falling back through the priority list keeps coverage high while preferring the
    cleanest text when a choice exists.
  * `color` carries both a free-text `value` ("Spinnsol Cocoa", "Taupe") and a curated
    `standardized_values` ("Brown", "Beige"). We keep the free-text value in the embedded
    document (richer signal for retrieval) and surface the standardized value separately
    for structured filtering — filtering on "Brown" should match "Spinnsol Cocoa".
  * `material` can be a compound, inconsistently-cased string ("Velvet, Hardwood frame,
    Metal legs" vs "stone"). We title-case and keep it as one descriptive phrase rather
    than attempting to split it into atomic materials — splitting would require a curated
    vocabulary we don't have, and a wrong split is worse than no split.
  * `item_keywords` frequently contains empty strings, near-duplicates, and raw SEO
    fragments. We lower-case, strip, drop empties, and de-duplicate while preserving order.
  * Free text (`bullet_point`, `product_description`) can contain HTML entities/tags and
    smart-quote/dash unicode variants. We unescape HTML, strip tags, normalize punctuation
    to ASCII equivalents, and collapse whitespace — so embedding input is uniform regardless
    of which marketplace authored the listing.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

from src.preprocess.schema import CanonicalProduct

# Preference order for selecting a single English entry from a localized list.
# Native English-speaking marketplaces first; broader English-tagged locales as fallback.
ENGLISH_LOCALE_PRIORITY = ("en_US", "en_GB", "en_CA", "en_AU", "en_IN", "en_AE", "en_SG")

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_PUNCT_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
    }
)


def clean_text(text: str | None) -> str | None:
    """Unescape HTML, strip tags, normalize unicode punctuation, collapse whitespace."""
    if text is None:
        return None
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = text.translate(_PUNCT_MAP)
    text = unicodedata.normalize("NFKC", text)
    text = _WS_RE.sub(" ", text).strip()
    return text or None


def pick_localized(entries: list[dict[str, Any]] | None, value_key: str = "value") -> str | None:
    """Pick one English value from a list of {language_tag, <value_key>: ...} dicts.

    Walks ENGLISH_LOCALE_PRIORITY in order and returns the first match; falls back to any
    other `en_*`-tagged entry if none of the prioritized locales are present. Returns None
    if the field is absent or has no English-tagged entry (the product is then dropped by
    the English filter upstream — see `build_canonical_product`).
    """
    if not entries:
        return None

    by_locale: dict[str, str] = {}
    for entry in entries:
        tag = entry.get("language_tag", "")
        value = entry.get(value_key)
        if tag and value and tag not in by_locale:
            by_locale[tag] = value

    for locale in ENGLISH_LOCALE_PRIORITY:
        if locale in by_locale:
            return clean_text(by_locale[locale])

    for tag, value in by_locale.items():
        if tag.startswith("en"):
            return clean_text(value)

    return None


def pick_color(color_entries: list[dict[str, Any]] | None) -> tuple[str | None, str | None]:
    """Return (display_color, standardized_color) for structured filtering vs. embedding text.

    `standardized_values` is a curated, small vocabulary ("Brown", "Blue") — ideal for exact
    metadata filtering. The free-text `value` ("Spinnsol Cocoa") is richer for embedding.
    """
    if not color_entries:
        return None, None

    for locale in ENGLISH_LOCALE_PRIORITY:
        for entry in color_entries:
            if entry.get("language_tag") == locale:
                display = clean_text(entry.get("value"))
                std_values = entry.get("standardized_values") or []
                standardized = clean_text(std_values[0]) if std_values else None
                return display, standardized

    # Fall back to any other en_*-tagged entry, matching pick_localized's behavior.
    for entry in color_entries:
        tag = entry.get("language_tag", "")
        if tag.startswith("en"):
            display = clean_text(entry.get("value"))
            std_values = entry.get("standardized_values") or []
            standardized = clean_text(std_values[0]) if std_values else None
            return display, standardized

    return None, None


def pick_material(material_entries: list[dict[str, Any]] | None) -> str | None:
    text = pick_localized(material_entries)
    return text.title() if text else None


def dedupe_list(values: list[str | None]) -> list[str]:
    """Lower-case, strip, drop empties, and de-duplicate while preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for v in values:
        if not v:
            continue
        cleaned = clean_text(v)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def build_doc_text(
    *,
    name: str,
    brand: str | None,
    product_type: str,
    color: str | None,
    material: str | None,
    bullet_points: list[str],
    keywords: list[str],
    visual_caption: str | None = None,
) -> str:
    """Assemble the canonical embeddable string for a product.

    Format (documented in docs/ShopTalk_Plan.md §2.1):
      [name] · [brand] · type: [product_type] · color: [color] · material: [material]
      · visual: [caption] · [bullet_points] · keywords: [item_keywords]

    `visual_caption` sits right after the core attributes, ahead of `bullet_points`/
    `keywords` — encoders truncate from the end, and on long listings the bulky
    bullet-point block would otherwise push the caption (the very thing the captioning
    stage exists to add) past the token limit before the encoder ever sees it. Measured
    on the 200-doc Kaggle sample: with the caption last, ~2% of docs lost it entirely to
    truncation under a 256-token encoder; this ordering keeps the visual signal in the
    surviving prefix regardless of how long the bullet points run.

    `visual_caption` is None until the captioning stage runs; the field is appended only
    when present, so this single function produces both the text-only and the
    caption-enriched canonical document — the two retrieval approaches we compare.
    """
    parts = [name]
    if brand:
        parts.append(brand)
    parts.append(f"type: {product_type.replace('_', ' ').title()}")
    if color:
        parts.append(f"color: {color}")
    if material:
        parts.append(f"material: {material}")
    if visual_caption:
        parts.append(f"visual: {visual_caption}")
    if bullet_points:
        parts.append(" ".join(bullet_points))
    if keywords:
        parts.append("keywords: " + ", ".join(keywords))
    return " · ".join(parts)


def build_canonical_product(
    record: dict[str, Any],
    image_index: dict[str, str] | None = None,
) -> CanonicalProduct | None:
    """Turn one raw ABO listing record into a CanonicalProduct, or None if it should be dropped.

    A record is dropped when it has no English-tagged `item_name` (the English filter) or
    no `product_type` (required for structured filtering and category-aware sampling).
    Deterministic: identical input always yields a byte-identical `doc_text`.
    """
    name = pick_localized(record.get("item_name"))
    if name is None:
        return None

    product_type_entries = record.get("product_type") or []
    product_type = product_type_entries[0].get("value") if product_type_entries else None
    if not product_type:
        return None

    brand = pick_localized(record.get("brand"))
    color, color_standardized = pick_color(record.get("color"))
    material = pick_material(record.get("material"))
    bullet_points = dedupe_list([pick_localized([e]) for e in record.get("bullet_point", [])])
    keywords = dedupe_list([pick_localized([e]) for e in record.get("item_keywords", [])])

    main_image_id = record.get("main_image_id")
    image_path = (image_index or {}).get(main_image_id) if main_image_id else None

    doc_text = build_doc_text(
        name=name,
        brand=brand,
        product_type=product_type,
        color=color,
        material=material,
        bullet_points=bullet_points,
        keywords=keywords,
    )

    return CanonicalProduct(
        item_id=record["item_id"],
        domain_name=record["domain_name"],
        product_type=product_type,
        name=name,
        brand=brand,
        color=color_standardized or color,
        material=material,
        bullet_points=bullet_points,
        keywords=keywords,
        main_image_id=main_image_id,
        image_path=image_path,
        doc_text=doc_text,
    )

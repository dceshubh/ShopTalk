"""Structured representation of a product — the contract between raw ABO records and
everything downstream (captioning, embedding, indexing, the API). Using Pydantic here
(rather than passing raw dicts around) means every consumer gets validation for free and
the canonical-document shape is documented in exactly one place.
"""

from __future__ import annotations

from pydantic import BaseModel


class CanonicalProduct(BaseModel):
    """One row of the cleaned, English, structured catalog — the unit we embed and index.

    `doc_text` is the exact string handed to the embedding encoder. Constructing it here
    (rather than inline at embedding time) is what guarantees the offline indexer and the
    online API represent a product identically.
    """

    item_id: str
    domain_name: str
    product_type: str
    name: str
    brand: str | None = None
    color: str | None = None
    material: str | None = None
    bullet_points: list[str] = []
    keywords: list[str] = []
    main_image_id: str | None = None
    image_path: str | None = None
    doc_text: str

    model_config = {"frozen": True}

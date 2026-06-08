"""Product-card lookup for the API — resolves the agent's retrieved `item_id`s into the
display fields (`name`, `image_path`, attributes) the UI renders as product cards.

Loaded ONCE at startup from the same `products_parquet` the offline indexer reads
(`paths.products_parquet` in config.yaml) — one more instance of "shared module, same
data, train↔inference parity," not a second hand-rolled lookup that could drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.common.config import load_config, resolve_path

_CARD_COLUMNS = ["item_id", "name", "brand", "color", "material", "product_type", "image_path"]


@dataclass
class ProductCatalog:
    by_id: dict[str, dict]

    def get(self, item_id: str) -> dict | None:
        return self.by_id.get(item_id)


def load_catalog(parquet_path: Path | str | None = None) -> ProductCatalog:
    path = Path(parquet_path) if parquet_path else resolve_path(load_config()["paths"]["products_parquet"])
    df = pd.read_parquet(path, columns=_CARD_COLUMNS)
    df = df.set_index("item_id")
    df = df.where(pd.notnull(df), None)  # NaN -> None so Pydantic sees a clean Optional[str]
    return ProductCatalog(by_id=df.to_dict(orient="index"))

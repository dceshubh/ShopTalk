"""Request/response schemas for the FastAPI inference service — Pydantic, not ad-hoc dicts,
so the OpenAPI schema is generated for free and a malformed payload fails validation with a
structured 422 rather than surfacing as an obscure downstream error."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, description="Stable identity across sessions — keys persistent memory")
    session_id: str = Field(
        min_length=1, description="Conversation identity — keys the in-RAM history buffer"
    )
    message: str = Field(min_length=1, description="The shopper's message for this turn")


class ProductCard(BaseModel):
    item_id: str
    name: str | None = None
    image_path: str | None = Field(default=None, description="Relative to the /images mount")
    product_type: str | None = None
    color: str | None = None
    material: str | None = None
    brand: str | None = None


class ChatResponse(BaseModel):
    response_text: str
    products: list[ProductCard] = Field(
        description="Sourced ENTIRELY from this turn's retrieval — see src.agent.graph for why "
        "this is what makes 'no hallucinated products' true by construction."
    )


class HealthResponse(BaseModel):
    status: str
    generator_model: str
    encoder_model: str
    collection: str
    catalog_size: int
    started_at: str
    load_count: int = Field(
        description="Stays at 1 for the process lifetime — proves models load ONCE at startup"
    )


class ErrorDetail(BaseModel):
    request_id: str
    message: str

"""The ShopTalk inference service (docs/ShopTalk_Plan.md §2.7 / Phase 6).

**Models load exactly ONCE at startup** — inside `lifespan`, never per request — and are
held in `app.state.models` for the process lifetime. `app.state.load_count` is the
exit-gate proof: it is incremented exactly once per app lifecycle and is exposed via
`/health`, so "a second request does not reload" is assertable from the outside, not just
claimed in a comment.

Endpoints:
    POST /chat              -> agent turn: response text + grounded product cards
    GET  /health            -> loaded-model identities + load_count + catalog size
    GET  /products/{id}     -> a single product card (404 if unknown)
    GET  /images/{path}     -> static product images (mounted from `captioning.images_cache_dir`)
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.agent.graph import ShoppingAgent, build_graph
from src.agent.llm import load_generator
from src.agent.memory import load_persistent_memory
from src.agent.personalize import Personalizer
from src.api.catalog import ProductCatalog, load_catalog
from src.api.schemas import ChatRequest, ChatResponse, ErrorDetail, HealthResponse, ProductCard
from src.common.config import load_config, resolve_path
from src.common.logging import get_logger
from src.common.timer import Timer
from src.embeddings.encode import load_encoder
from src.index.build import load_collection
from src.ui.feedback import load_feedback_store

logger = get_logger(__name__)


@dataclass
class RuntimeModels:
    """Everything the request handlers need — built once by a `loader` and frozen into
    `app.state.models`. Splitting this out (rather than loading inline in `lifespan`) is
    what lets tests substitute a fake loader without booting the Groq client, the encoder,
    or Chroma."""

    agent: ShoppingAgent
    catalog: ProductCatalog
    generator_model: str
    encoder_model: str
    collection_name: str


def _load_real_models() -> RuntimeModels:
    cfg = load_config()
    encoder_model = cfg["models"]["text_encoder"]["primary"]

    llm = load_generator()
    encoder = load_encoder(encoder_model)
    collection = load_collection(encoder_model, "caption_enriched")
    persistent_memory = load_persistent_memory()
    personalizer = Personalizer(feedback_store=load_feedback_store())
    catalog = load_catalog()

    top_k = cfg["retrieval"]["top_k"]
    graph = build_graph(
        llm,
        collection,
        encoder,
        personalizer,
        persistent_memory,
        top_k=top_k,
        pool_size=cfg["retrieval"]["personalization_pool_size"],
    )
    agent = ShoppingAgent(graph=graph, persistent_memory=persistent_memory)

    return RuntimeModels(
        agent=agent,
        catalog=catalog,
        generator_model=llm.model_name,
        encoder_model=encoder_model,
        collection_name=collection.name,
    )


def _build_lifespan(loader: Callable[[], RuntimeModels]):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Loading models for the API — this happens exactly ONCE per process")
        with Timer("api.startup.load_models"):
            app.state.models = loader()
        app.state.load_count = getattr(app.state, "load_count", 0) + 1
        app.state.started_at = datetime.now(UTC).isoformat()
        logger.info(
            "Startup complete: load_count=%d generator=%s encoder=%s collection=%s catalog_size=%d",
            app.state.load_count,
            app.state.models.generator_model,
            app.state.models.encoder_model,
            app.state.models.collection_name,
            len(app.state.models.catalog.by_id),
        )
        yield
        logger.info("Shutting down — models held for the process lifetime, nothing to unload")

    return lifespan


def _to_card(catalog: ProductCatalog, item_id: str) -> ProductCard | None:
    row = catalog.get(item_id)
    if row is None:
        return None
    return ProductCard(item_id=item_id, **row)


def create_app(*, loader: Callable[[], RuntimeModels] = _load_real_models) -> FastAPI:
    """App factory — production (`app = create_app()`) and tests (`create_app(loader=fake)`)
    share every route/middleware/handler and differ only in how models get built."""
    app = FastAPI(
        title="ShopTalk",
        description="Conversational shopping assistant API",
        lifespan=_build_lifespan(loader),
    )

    images_dir = resolve_path(load_config()["captioning"]["images_cache_dir"])
    if images_dir.is_dir():
        app.mount("/images", StaticFiles(directory=images_dir), name="images")

    # ------------------------------------------------------------------
    # Request context: a request_id on every response/log line, and a
    # per-request latency measurement (the same Timer used everywhere else
    # in the project — "every latency number measured the same way").
    # ------------------------------------------------------------------

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %d (%.2f ms) [request_id=%s]",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response

    # ------------------------------------------------------------------
    # Structured error handling — bad input is a structured 4xx body with
    # a request_id to correlate against logs, never a raw 500 stacktrace.
    # ------------------------------------------------------------------

    def _error_response(request: Request, status_code: int, message: str, **extra) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=status_code,
            content=ErrorDetail(request_id=request_id, message=message).model_dump() | extra,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(
            "Validation error [request_id=%s]: %s",
            getattr(request.state, "request_id", "unknown"),
            exc.errors(),
        )
        return _error_response(
            request, 422, "Invalid request payload — see `errors` for details.", errors=exc.errors()
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return _error_response(request, exc.status_code, str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error [request_id=%s]", getattr(request.state, "request_id", "unknown"))
        return _error_response(request, 500, "Internal error — include this request_id when reporting.")

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        models: RuntimeModels = request.app.state.models
        return HealthResponse(
            status="ok",
            generator_model=models.generator_model,
            encoder_model=models.encoder_model,
            collection=models.collection_name,
            catalog_size=len(models.catalog.by_id),
            started_at=request.app.state.started_at,
            load_count=request.app.state.load_count,
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        models: RuntimeModels = request.app.state.models
        request_id = request.state.request_id
        with Timer(f"api.chat [request_id={request_id}, session={payload.session_id}]"):
            turn = models.agent.chat(
                user_id=payload.user_id, session_id=payload.session_id, message=payload.message
            )
        cards = [
            card for item_id in turn.product_ids if (card := _to_card(models.catalog, item_id)) is not None
        ]
        return ChatResponse(response_text=turn.response_text, products=cards)

    @app.get("/products/{item_id}", response_model=ProductCard)
    async def get_product(item_id: str, request: Request) -> ProductCard:
        models: RuntimeModels = request.app.state.models
        card = _to_card(models.catalog, item_id)
        if card is None:
            raise HTTPException(status_code=404, detail=f"Unknown product id {item_id!r}")
        return card

    return app


app = create_app()

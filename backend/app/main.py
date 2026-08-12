"""FastAPI application.

Startup wires the whole object graph once — retriever, claim store, tools,
chat model, agent — and hangs it off ``app.state``. Building it per request
would re-open Chroma and re-read the policy document on every message.

A missing or invalid API key does **not** crash the process. The app starts,
``/health`` answers, and ``/chat`` returns 503 with the exact variable to set.
A container that boots and explains itself is far easier to diagnose than one
in a crash loop, and it keeps ``docker compose up`` from looking broken when
the only problem is an empty ``.env``.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent.graph import PolicyAgent
from app.agent.llm import build_chat_model
from app.agent.tools import build_tools
from app.api.v1.routes import router as v1_router
from app.config import Settings, get_settings
from app.rag.retriever import PolicyRetriever
from app.tools.store import ClaimStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("omnicare")

REQUEST_ID_HEADER = "X-Request-ID"


def _bootstrap(app: FastAPI, settings: Settings) -> None:
    """Build the agent and index the policy document."""
    retriever = PolicyRetriever(
        persist_dir=settings.chroma_dir,
        collection_name=settings.collection_name,
        embedding_backend=settings.embedding_backend,
        top_k=settings.retrieval_top_k,
        max_chunk_chars=settings.max_chunk_chars,
    )
    indexed = retriever.ingest_file(settings.policy_path)
    app.state.indexed_chunks = indexed
    logger.info("Indexed %d policy chunks from %s", indexed, settings.policy_path)

    store = ClaimStore(settings.claims_path, lock_timeout=settings.file_lock_timeout)
    tools = build_tools(
        retriever=retriever, store=store, top_k=settings.retrieval_top_k
    )
    model = build_chat_model(
        provider=settings.llm_provider,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        groq_api_key=settings.groq_api_key,
        openai_api_key=settings.openai_api_key,
        anthropic_api_key=settings.anthropic_api_key,
    )
    app.state.agent = PolicyAgent(
        chat_model=model,
        tools=tools,
        max_iterations=settings.max_tool_iterations,
        max_message_chars=settings.max_message_chars,
    )
    logger.info("Agent ready on provider %s", settings.llm_provider)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = getattr(app.state, "settings", None) or get_settings()
    app.state.settings = settings
    app.state.agent = None
    app.state.indexed_chunks = 0
    app.state.startup_error = ""

    try:
        _bootstrap(app, settings)
    except Exception as exc:  # noqa: BLE001 - reported through /chat and /ready
        app.state.startup_error = str(exc)
        logger.exception("Startup failed, /chat will return 503: %s", exc)

    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory. Tests pass their own settings or override state."""
    app = FastAPI(
        title="OmniCare Financial Customer Assistant",
        version="1.0.0",
        description=(
            "Policy coverage answers with citations, claim status lookups, and "
            "claim submission."
        ),
        lifespan=lifespan,
    )
    if settings is not None:
        app.state.settings = settings

    # The frontend is a separate origin in Compose; this is a local prototype,
    # so the allowance is broad. A deployment would name its origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def tag_and_time(request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "%s %s -> %d in %.0fms [%s]",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Last resort. The traceback goes to the log, never to the client."""
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Something went wrong. Please try again."},
        )

    app.include_router(v1_router)
    return app


app = create_app()

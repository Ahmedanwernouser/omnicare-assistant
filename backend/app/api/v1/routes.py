"""Versioned API routes.

The two endpoints named in the specification return exactly the documented
shape and nothing more — no request IDs, no timings, no debug fields — so a
client written against the spec keeps working. Diagnostics travel in headers
and logs instead.

``/ready`` is additional rather than a modification: ``/health`` answers "is
the process alive", which is what it is specified to do, while Compose needs
"is the vector store loaded and the model configured" before it starts the
frontend. Overloading ``/health`` with that would have broken the contract.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.agent.graph import PolicyAgent
from app.agent.llm import MissingAPIKeyError
from app.agent.prompts import UPSTREAM_ERROR_FALLBACK
from app.schemas import ChatRequest, ChatResponse, HealthResponse, ToolCall

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


def get_agent(request: Request) -> PolicyAgent:
    """Resolve the agent built during startup."""
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        detail = getattr(request.app.state, "startup_error", "") or (
            "The assistant is not initialised."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail
        )
    return agent


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness. Exactly ``{"status": "healthy"}``."""
    return HealthResponse()


@router.get("/ready", response_model=HealthResponse)
def ready(request: Request) -> HealthResponse:
    """Readiness: the agent is built and the policy index is populated."""
    agent = getattr(request.app.state, "agent", None)
    indexed = getattr(request.app.state, "indexed_chunks", 0)
    if agent is None or indexed == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=getattr(request.app.state, "startup_error", "")
            or "The policy index is empty.",
        )
    return HealthResponse(status="ready")


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, agent: PolicyAgent = Depends(get_agent)) -> ChatResponse:
    """Handle one conversational turn.

    A refusal is a normal outcome and returns 200 with the refusal text — the
    turn succeeded, the answer was "no". Only infrastructure failures use an
    error status, so a client can tell "the assistant declined" apart from
    "the assistant is down".
    """
    try:
        result = agent.run(user_id=payload.user_id, message=payload.message)
    except MissingAPIKeyError as exc:
        logger.error("Chat failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001 - mapped below, always logged
        logger.exception("Chat turn failed for user %s", payload.user_id)
        raise HTTPException(
            status_code=_status_for(exc), detail=_detail_for(exc)
        ) from exc

    return ChatResponse(
        response=result.response,
        sources=result.sources,
        tool_calls=[ToolCall(**call) for call in result.tool_calls],
    )


def _status_for(exc: Exception) -> int:
    """Rate limits and provider outages are upstream problems, not bugs."""
    text = f"{type(exc).__name__} {exc}".lower()
    if any(
        marker in text
        for marker in ("rate limit", "ratelimit", "429", "quota", "too many requests")
    ):
        return status.HTTP_429_TOO_MANY_REQUESTS
    if any(
        marker in text
        for marker in ("timeout", "connection", "unavailable", "503", "502")
    ):
        return status.HTTP_503_SERVICE_UNAVAILABLE
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _detail_for(exc: Exception) -> str:
    """Never leak an internal traceback to a policyholder."""
    code = _status_for(exc)
    if code == status.HTTP_429_TOO_MANY_REQUESTS:
        return (
            "The assistant is rate limited by the model provider right now. "
            "Please wait a moment and try again."
        )
    if code == status.HTTP_503_SERVICE_UNAVAILABLE:
        return UPSTREAM_ERROR_FALLBACK
    return "Something went wrong handling that message. Please try again."

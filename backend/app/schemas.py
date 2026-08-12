"""Public API contract.

The field names here are fixed by the assessment specification and are not
negotiable: ``/api/v1/chat`` accepts ``{user_id, message}`` and returns
``{response, sources, tool_calls}``. Nothing is added to the top level of the
response, so a client written against the spec keeps working.

``sources`` is deliberately ``list[str]`` rather than a list of objects — the
spec shows ``["..."]``. Each entry is a self-describing citation string, e.g.
``sample_policy.md § Section 1: Home Water Damage Coverage``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

#: Hard transport ceiling. The *business* limit on message length is
#: ``Settings.max_message_chars`` and is enforced by the agent's guard node;
#: this larger bound simply stops a multi-megabyte body from reaching it.
MAX_REQUEST_CHARS = 8000


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    user_id: str = Field(min_length=1, max_length=64, examples=["usr_123"])
    message: str = Field(
        min_length=1,
        max_length=MAX_REQUEST_CHARS,
        examples=["Is water damage from a burst pipe covered?"],
    )


class ToolCall(BaseModel):
    """One tool invocation, surfaced so the UI can show what the agent did."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    summary: str = ""


class ChatResponse(BaseModel):
    response: str
    sources: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Exactly ``{"status": "healthy"}`` — no extra keys."""

    status: str = "healthy"


class ErrorResponse(BaseModel):
    detail: str

"""Backend client for the Streamlit UI.

Kept apart from ``app.py`` so the network handling can be tested without
booting Streamlit. Everything here returns plain data; nothing draws.

Error strings are part of the product, not debug output. Each one names what
happened and what to do next — the environment variable to set, the command to
run — because the person reading it is usually a reviewer three minutes into
their first ``docker compose up``.
"""

from __future__ import annotations

import os

import requests

#: Inside Compose the backend is reachable by service name, not localhost.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
CHAT_ENDPOINT = f"{BACKEND_URL}/api/v1/chat"
HEALTH_ENDPOINT = f"{BACKEND_URL}/api/v1/health"
READY_ENDPOINT = f"{BACKEND_URL}/api/v1/ready"

HEALTH_TIMEOUT = 5
CHAT_TIMEOUT = 90

STATUS_OK = "ok"
STATUS_STARTING = "starting"
STATUS_DOWN = "down"


def backend_status() -> tuple[str, str]:
    """Return ``(state, detail)`` where state is ok, starting or down."""
    try:
        health = requests.get(HEALTH_ENDPOINT, timeout=HEALTH_TIMEOUT)
        if health.status_code != 200:
            return STATUS_DOWN, "The backend answered, but not with a healthy status."

        ready = requests.get(READY_ENDPOINT, timeout=HEALTH_TIMEOUT)
        if ready.status_code == 200:
            return STATUS_OK, ""
        return STATUS_STARTING, detail_of(
            ready, "The policy index is still loading."
        )
    except requests.RequestException:
        return STATUS_DOWN, (
            f"No response from {BACKEND_URL}. Start it with `docker compose up`, "
            "or point BACKEND_URL at the API."
        )


def ask(user_id: str, message: str) -> dict:
    """Send one turn.

    Returns the API body on success, or ``{"error": "..."}`` with a message
    that is safe and useful to show the user.
    """
    try:
        response = requests.post(
            CHAT_ENDPOINT,
            json={"user_id": user_id, "message": message},
            timeout=CHAT_TIMEOUT,
        )
    except requests.Timeout:
        return {"error": "The assistant took too long to answer. Try asking again."}
    except requests.RequestException:
        return {
            "error": (
                f"Can't reach the backend at {BACKEND_URL}. "
                "Check that it's running and try again."
            )
        }

    if response.status_code == 200:
        return _validate(response)
    if response.status_code == 422:
        return {"error": "That message couldn't be sent. Try a shorter one."}
    if response.status_code == 429:
        return {
            "error": detail_of(
                response, "The model provider is rate limiting us. Wait a moment."
            )
        }
    if response.status_code == 503:
        return {"error": detail_of(response, "The assistant isn't ready yet.")}
    return {"error": detail_of(response, "Something went wrong. Try again.")}


def _validate(response: requests.Response) -> dict:
    """Coerce the body into the shape the UI renders.

    A 200 with an unexpected body should show a clear message rather than
    raise inside a render loop.
    """
    try:
        body = response.json()
    except ValueError:
        return {"error": "The backend sent a reply that couldn't be read."}
    if not isinstance(body, dict) or "response" not in body:
        return {"error": "The backend sent a reply that couldn't be read."}
    return {
        "response": str(body.get("response", "")),
        "sources": list(body.get("sources") or []),
        "tool_calls": list(body.get("tool_calls") or []),
    }


def detail_of(response: requests.Response, fallback: str) -> str:
    """Prefer the API's own explanation; fall back to ours."""
    try:
        body = response.json()
    except ValueError:
        return fallback
    if isinstance(body, dict) and body.get("detail"):
        return str(body["detail"])
    return fallback

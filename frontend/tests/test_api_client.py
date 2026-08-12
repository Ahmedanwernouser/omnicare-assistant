"""Frontend backend-client behaviour.

These run against a real FastAPI app served in-process by ``requests_mock``-
style monkeypatching of ``requests``, so no server and no network are needed.
What matters is that every response the backend can produce turns into
something useful on screen — an answer, or a sentence telling the user what to
do next.
"""

from __future__ import annotations

import pytest
import requests

import api_client
from api_client import (
    STATUS_DOWN,
    STATUS_OK,
    STATUS_STARTING,
    ask,
    backend_status,
    detail_of,
)


class FakeResponse:
    def __init__(self, status_code: int, body=None, text: str = "") -> None:
        self.status_code = status_code
        self._body = body
        self._text = text

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


@pytest.fixture
def post(monkeypatch):
    """Capture the outgoing POST and control the reply."""
    sent = {}

    def _install(response):
        def _post(url, json=None, timeout=None):
            sent["url"] = url
            sent["json"] = json
            sent["timeout"] = timeout
            if isinstance(response, Exception):
                raise response
            return response

        monkeypatch.setattr(requests, "post", _post)
        return sent

    return _install


@pytest.fixture
def get(monkeypatch):
    def _install(health, ready=None):
        def _get(url, timeout=None):
            if isinstance(health, Exception):
                raise health
            return ready if url.endswith("/ready") and ready is not None else health

        monkeypatch.setattr(requests, "get", _get)

    return _install


# -- successful turn -------------------------------------------------------


def test_successful_turn_is_passed_through(post) -> None:
    body = {
        "response": "Covered up to $25,000.",
        "sources": ["sample_policy.md § Section 1: Home Water Damage Coverage"],
        "tool_calls": [{"name": "search_policy", "ok": True}],
    }
    sent = post(FakeResponse(200, body))
    result = ask("usr_1", "Is a burst pipe covered?")

    assert result["response"] == "Covered up to $25,000."
    assert result["sources"] == body["sources"]
    assert sent["json"] == {"user_id": "usr_1", "message": "Is a burst pipe covered?"}


def test_missing_lists_become_empty_not_none(post) -> None:
    """The UI iterates these; None would raise inside a render loop."""
    post(FakeResponse(200, {"response": "Hello."}))
    result = ask("usr_1", "Hi")
    assert result["sources"] == []
    assert result["tool_calls"] == []


def test_null_lists_become_empty(post) -> None:
    post(FakeResponse(200, {"response": "Hi", "sources": None, "tool_calls": None}))
    result = ask("usr_1", "Hi")
    assert result["sources"] == []


# -- error responses become directions -------------------------------------


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (422, {"detail": "too long"}, "shorter"),
        (429, {"detail": "Rate limited by the provider."}, "Rate limited"),
        (503, {"detail": "GROQ_API_KEY is not set."}, "GROQ_API_KEY"),
        (500, {"detail": "Something went wrong."}, "went wrong"),
    ],
)
def test_error_statuses_produce_actionable_text(
    post, status: int, body: dict, expected: str
) -> None:
    post(FakeResponse(status, body))
    result = ask("usr_1", "hello")

    assert "error" in result
    assert expected in result["error"]
    assert "response" not in result


def test_503_surfaces_the_variable_to_set(post) -> None:
    """The single most common first-run failure must name its own fix."""
    post(FakeResponse(503, {"detail": "GROQ_API_KEY is not set. Copy .env.example."}))
    assert "GROQ_API_KEY" in ask("usr_1", "hi")["error"]


def test_backend_unreachable_names_the_url_and_the_command(post) -> None:
    post(requests.ConnectionError("refused"))
    error = ask("usr_1", "hi")["error"]
    assert api_client.BACKEND_URL in error
    assert "running" in error


def test_timeout_suggests_retrying(post) -> None:
    post(requests.Timeout("slow"))
    assert "too long" in ask("usr_1", "hi")["error"]


def test_unreadable_success_body_does_not_raise(post) -> None:
    post(FakeResponse(200, None))
    assert "couldn't be read" in ask("usr_1", "hi")["error"]


def test_200_with_the_wrong_shape_is_caught(post) -> None:
    post(FakeResponse(200, {"unexpected": "shape"}))
    assert "error" in ask("usr_1", "hi")


def test_error_body_without_detail_falls_back(post) -> None:
    post(FakeResponse(503, {}))
    assert ask("usr_1", "hi")["error"] == "The assistant isn't ready yet."


# -- status probe ----------------------------------------------------------


def test_status_ok_when_healthy_and_ready(get) -> None:
    get(FakeResponse(200, {"status": "healthy"}), FakeResponse(200, {"status": "ready"}))
    assert backend_status() == (STATUS_OK, "")


def test_status_starting_while_the_index_loads(get) -> None:
    get(
        FakeResponse(200, {"status": "healthy"}),
        FakeResponse(503, {"detail": "The policy index is empty."}),
    )
    state, detail = backend_status()
    assert state == STATUS_STARTING
    assert "index" in detail


def test_status_down_when_nothing_answers(get) -> None:
    get(requests.ConnectionError("refused"))
    state, detail = backend_status()
    assert state == STATUS_DOWN
    assert "docker compose up" in detail


def test_status_down_on_an_unhealthy_reply(get) -> None:
    get(FakeResponse(500, {}))
    assert backend_status()[0] == STATUS_DOWN


# -- helpers ---------------------------------------------------------------


def test_detail_prefers_the_api_explanation() -> None:
    assert detail_of(FakeResponse(503, {"detail": "real reason"}), "fallback") == (
        "real reason"
    )


def test_detail_falls_back_on_non_json() -> None:
    assert detail_of(FakeResponse(503, None), "fallback") == "fallback"


def test_chat_timeout_is_generous_enough_for_a_tool_loop() -> None:
    """A multi-tool turn on a free-tier model is not fast."""
    assert api_client.CHAT_TIMEOUT >= 60
    assert api_client.HEALTH_TIMEOUT <= 10

"""Chat model factory: provider selection, defaults and credential errors."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.llm import (
    DEFAULT_MODELS,
    MissingAPIKeyError,
    ScriptedChatModel,
    build_chat_model,
)


def test_scripted_provider_needs_no_credentials() -> None:
    model = build_chat_model(provider="scripted")
    assert isinstance(model, ScriptedChatModel)


@pytest.mark.parametrize(
    ("provider", "env_name"),
    [
        ("groq", "GROQ_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
    ],
)
def test_missing_key_says_which_variable_to_set(provider: str, env_name: str) -> None:
    """The error a reviewer hits on first run must tell them what to do."""
    with pytest.raises(MissingAPIKeyError) as exc:
        build_chat_model(provider=provider)
    assert env_name in str(exc.value)
    assert ".env" in str(exc.value)


def test_unknown_provider_lists_the_valid_ones() -> None:
    with pytest.raises(ValueError) as exc:
        build_chat_model(provider="gemini")
    message = str(exc.value)
    assert "gemini" in message
    for known in DEFAULT_MODELS:
        assert known in message


def test_provider_name_is_case_and_space_insensitive() -> None:
    assert isinstance(build_chat_model(provider="  SCRIPTED "), ScriptedChatModel)


def test_every_real_provider_has_a_default_model() -> None:
    assert set(DEFAULT_MODELS) == {"groq", "openai", "anthropic"}
    assert all(name for name in DEFAULT_MODELS.values())


# -- the scripted model itself --------------------------------------------


def test_scripted_model_replays_in_order() -> None:
    model = ScriptedChatModel(
        responses=[AIMessage(content="first"), AIMessage(content="second")]
    )
    assert model.invoke([HumanMessage(content="a")]).content == "first"
    assert model.invoke([HumanMessage(content="b")]).content == "second"


def test_scripted_model_repeats_its_last_response() -> None:
    """A mis-scripted test should fail on an assertion, not hang."""
    model = ScriptedChatModel(responses=[AIMessage(content="only")])
    for _ in range(3):
        assert model.invoke([HumanMessage(content="x")]).content == "only"


def test_scripted_model_returns_a_fresh_object_each_call() -> None:
    """Reusing one message object makes add_messages replace instead of append."""
    model = ScriptedChatModel(responses=[AIMessage(content="hi")])
    first = model.invoke([HumanMessage(content="a")])
    second = model.invoke([HumanMessage(content="b")])
    assert first is not second


def test_scripted_model_records_what_it_was_asked() -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="ok")])
    model.invoke([HumanMessage(content="the question")])
    assert "the question" in str(model.calls[0][0].content)


def test_bind_tools_records_the_tools_and_stays_bindable() -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="ok")])
    bound = model.bind_tools(["a", "b"])
    assert bound is model
    assert model.bound_tools == ["a", "b"]

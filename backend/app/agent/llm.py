"""Chat model construction.

One factory, four backends. Three are real providers; the fourth,
``ScriptedChatModel``, is what lets the agent graph be tested without a network
call or an API key.

Testing an agent is awkward precisely because the interesting logic — routing,
the tool loop, source accumulation, the iteration cap — sits *between* model
calls. Mocking at the HTTP layer is brittle and mocking the whole graph tests
nothing. A model that replays a fixed script of ``AIMessage`` objects, tool
calls and all, puts the seam in exactly the right place: every node and edge
runs for real, and only the token generation is stubbed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

logger = logging.getLogger(__name__)

#: Sensible free-tier defaults. Override with LLM_MODEL.
DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
}


class MissingAPIKeyError(RuntimeError):
    """Raised when the selected provider has no credentials configured."""


class ScriptedChatModel(BaseChatModel):
    """Replays a fixed list of ``AIMessage`` objects, one per invocation.

    ``bind_tools`` records the tools and returns ``self`` — the graph needs a
    bindable model, but a scripted model does not choose tools, the script
    does. The last message repeats if the graph asks for more turns than the
    script provides, which keeps a mis-scripted test from hanging.
    """

    # Pydantic (BaseChatModel is a Pydantic model) deep-copies defaults per
    # instance, so these are not shared class state. Declared with Field for
    # clarity rather than relying on the reader knowing that.
    responses: list[AIMessage] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedChatModel:
        self.bound_tools = list(tools)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if not self.responses:
            reply = AIMessage(content="")
        else:
            index = min(len(self.calls) - 1, len(self.responses) - 1)
            # A fresh copy per call, exactly as a real provider returns. Handing
            # back the same object twice makes LangChain stamp it with one id,
            # and ``add_messages`` then *replaces* the earlier turn instead of
            # appending — which would quietly collapse any multi-step test.
            reply = self.responses[index].model_copy(deep=True)
            reply.id = None
        return ChatResult(generations=[ChatGeneration(message=reply)])


def build_chat_model(
    *,
    provider: str,
    model: str = "",
    temperature: float = 0.0,
    groq_api_key: str = "",
    openai_api_key: str = "",
    anthropic_api_key: str = "",
) -> BaseChatModel:
    """Return a chat model for ``provider``.

    Provider packages are imported lazily so that installing only the one you
    use is enough, and so the test suite never imports an SDK it will not call.
    """
    name = provider.strip().lower()
    model_name = model.strip() or DEFAULT_MODELS.get(name, "")

    if name == "scripted":
        return ScriptedChatModel()

    if name == "groq":
        _require(groq_api_key, "GROQ_API_KEY")
        from langchain_groq import ChatGroq

        return ChatGroq(model=model_name, temperature=temperature, api_key=groq_api_key)

    if name == "openai":
        _require(openai_api_key, "OPENAI_API_KEY")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name, temperature=temperature, api_key=openai_api_key
        )

    if name == "anthropic":
        _require(anthropic_api_key, "ANTHROPIC_API_KEY")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_name, temperature=temperature, api_key=anthropic_api_key
        )

    raise ValueError(
        f"Unknown LLM provider {provider!r}. "
        f"Expected one of: {', '.join(sorted(DEFAULT_MODELS))}, scripted."
    )


def _require(key: str, env_name: str) -> None:
    if not key.strip():
        raise MissingAPIKeyError(
            f"{env_name} is not set. Copy .env.example to .env and add a key, "
            "or set LLM_PROVIDER to a provider you have credentials for."
        )

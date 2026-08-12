"""The agent graph.

::

    START ──► guard ──(blocked)──► refuse ──► END
                │
                └──(allowed)──► agent ◄──┐
                                  │      │
                                  ├──────┘ tools
                                  │
                                  └──► finalize ──► END

Written as an explicit ``StateGraph`` rather than with ``create_react_agent``
for three reasons, all of which show up in the requirements:

* **The guard needs to be a node.** Screening has to happen before the model
  sees the message and has to be able to end the turn on its own. A prebuilt
  agent gives no clean place for that.
* **``sources`` and ``tool_calls`` are part of the API contract.** A custom tool
  node captures citations as structured data at the moment of retrieval, so the
  response reports what actually happened instead of what the model wrote about
  what happened.
* **The README needs a diagram.** The picture above and the code below are the
  same object, so the documentation cannot quietly drift from the behaviour.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agent.guards import response_leaks_canary, screen_user_message
from app.agent.prompts import (
    CANARY_LEAK_FALLBACK,
    SYSTEM_PROMPT,
    refusal_for,
)
from app.agent.tools import AgentTools, ToolOutcome

logger = logging.getLogger(__name__)

ITERATION_CAP_MESSAGE = (
    "I wasn't able to finish that in a reasonable number of steps. Could you "
    "break it into one question at a time?"
)

EMPTY_ANSWER_FALLBACK = (
    "I don't have an answer for that. I can help with policy coverage, claim "
    "status, and filing a new claim."
)


class AgentState(TypedDict):
    """Per-turn graph state.

    ``messages`` accumulates across turns via the checkpointer; ``sources``,
    ``tool_calls`` and ``iterations`` are reset on every invocation because
    they describe *this* turn and go straight into the API response.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    sources: list[str]
    tool_calls: list[dict[str, Any]]
    iterations: int
    blocked_rule: str


@dataclass
class AgentResult:
    """Exactly what ``/api/v1/chat`` returns."""

    response: str
    sources: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class PolicyAgent:
    """Compiled graph plus the per-user conversation store."""

    def __init__(
        self,
        *,
        chat_model: BaseChatModel,
        tools: AgentTools,
        max_iterations: int = 5,
        max_message_chars: int = 2000,
    ) -> None:
        self._tools = tools
        self._max_iterations = max_iterations
        self._max_message_chars = max_message_chars
        self._model = chat_model.bind_tools(tools.schemas)
        self._checkpointer = MemorySaver()
        self._graph = self._build().compile(checkpointer=self._checkpointer)

    # -- public ------------------------------------------------------------

    def run(self, *, user_id: str, message: str) -> AgentResult:
        """Handle one user turn."""
        final = self._graph.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "sources": [],
                "tool_calls": [],
                "iterations": 0,
                "blocked_rule": "",
            },
            config={"configurable": {"thread_id": user_id}},
        )
        return AgentResult(
            response=_last_text(final["messages"]),
            sources=_attribute_sources(
                _last_text(final["messages"]), _dedupe(final.get("sources", []))
            ),
            tool_calls=list(final.get("tool_calls", [])),
        )

    @property
    def graph(self):
        """The compiled graph, exposed for tests and diagram generation."""
        return self._graph

    # -- construction ------------------------------------------------------

    def _build(self) -> StateGraph:
        builder = StateGraph(AgentState)
        builder.add_node("guard", self._guard_node)
        builder.add_node("refuse", self._refuse_node)
        builder.add_node("agent", self._agent_node)
        builder.add_node("tools", self._tools_node)
        builder.add_node("finalize", self._finalize_node)

        builder.add_edge(START, "guard")
        builder.add_conditional_edges(
            "guard", self._after_guard, {"refuse": "refuse", "agent": "agent"}
        )
        builder.add_edge("refuse", END)
        builder.add_conditional_edges(
            "agent", self._after_agent, {"tools": "tools", "finalize": "finalize"}
        )
        builder.add_edge("tools", "agent")
        builder.add_edge("finalize", END)
        return builder

    # -- nodes -------------------------------------------------------------

    def _guard_node(self, state: AgentState) -> dict[str, Any]:
        message = _last_human_text(state["messages"])
        verdict = screen_user_message(message, max_chars=self._max_message_chars)
        return {"blocked_rule": "" if verdict.allowed else verdict.rule}

    def _refuse_node(self, state: AgentState) -> dict[str, Any]:
        rule = state.get("blocked_rule", "")
        logger.info("Turn refused by guard rule %s", rule or "unknown")
        return {"messages": [AIMessage(content=refusal_for(rule))]}

    def _agent_node(self, state: AgentState) -> dict[str, Any]:
        conversation = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        reply = self._model.invoke(conversation)
        return {"messages": [reply]}

    def _tools_node(self, state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        requested = getattr(last, "tool_calls", None) or []

        messages: list[BaseMessage] = []
        sources = list(state.get("sources", []))
        recorded = list(state.get("tool_calls", []))

        for call in requested:
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            outcome = self._execute(name, args)

            messages.append(
                ToolMessage(
                    content=outcome.content,
                    tool_call_id=call.get("id", ""),
                    name=name,
                )
            )
            sources.extend(outcome.sources)
            recorded.append(
                {
                    "name": name,
                    "arguments": args,
                    "ok": outcome.ok,
                    "summary": outcome.summary,
                }
            )

        return {
            "messages": messages,
            "sources": sources,
            "tool_calls": recorded,
            "iterations": state.get("iterations", 0) + 1,
        }

    def _finalize_node(self, state: AgentState) -> dict[str, Any]:
        # Deliberately the last *AI* message, not the last message: after a tool
        # round the tail of the list is a ToolMessage, and reading that would
        # silently finalise on tool output instead of the model's answer.
        last = _last_ai(state["messages"])
        text = _text_of(last) if last is not None else ""

        # Reached only when the iteration cap stopped a pending tool call.
        if last is not None and getattr(last, "tool_calls", None) and not text.strip():
            logger.warning("Iteration cap reached with tool calls outstanding")
            return {"messages": [AIMessage(content=ITERATION_CAP_MESSAGE)]}

        if response_leaks_canary(text):
            logger.error("Response echoed the system prompt canary; suppressed")
            return {"messages": [AIMessage(content=CANARY_LEAK_FALLBACK)]}

        if not text.strip():
            return {"messages": [AIMessage(content=EMPTY_ANSWER_FALLBACK)]}

        return {}

    # -- routing -----------------------------------------------------------

    @staticmethod
    def _after_guard(state: AgentState) -> str:
        return "refuse" if state.get("blocked_rule") else "agent"

    def _after_agent(self, state: AgentState) -> str:
        last = _last_ai(state["messages"])
        if last is None or not getattr(last, "tool_calls", None):
            return "finalize"
        if state.get("iterations", 0) >= self._max_iterations:
            return "finalize"
        return "tools"

    # -- helpers -----------------------------------------------------------

    def _execute(self, name: str, args: dict[str, Any]) -> ToolOutcome:
        runner = self._tools.runners.get(name)
        if runner is None:
            logger.warning("Model requested unknown tool %r", name)
            return ToolOutcome(
                content=f"ERROR: no tool named {name!r} exists.",
                ok=False,
                summary=f"Unknown tool {name!r}.",
            )
        try:
            return runner(**args)
        except TypeError as exc:
            # Hallucinated or missing argument names land here.
            logger.warning("Bad arguments for %s: %s", name, exc)
            return ToolOutcome(
                content=f"ERROR: {name} was called with invalid arguments ({exc}).",
                ok=False,
                summary=f"Invalid arguments for {name}.",
            )
        except Exception as exc:  # noqa: BLE001 - a tool must never kill the turn
            logger.exception("Tool %s raised", name)
            return ToolOutcome(
                content=f"ERROR: {name} failed ({type(exc).__name__}).",
                ok=False,
                summary=f"{name} failed.",
            )


# --------------------------------------------------------------------------
# Message helpers
# --------------------------------------------------------------------------


def _text_of(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    # Some providers return content as a list of blocks.
    return "".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _last_ai(messages: list[BaseMessage]) -> AIMessage | None:
    """The most recent assistant message, ignoring any trailing tool output."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _last_text(messages: list[BaseMessage]) -> str:
    last = _last_ai(messages)
    return _text_of(last).strip() if last is not None else ""


def _last_human_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _text_of(message)
    return ""


def _dedupe(items: list[str]) -> list[str]:
    """Unique, order-preserving. The same section retrieved twice is cited once."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _attribute_sources(response: str, retrieved: list[str]) -> list[str]:
    """Narrow the retrieved citations to the ones the answer actually used.

    Retrieval returns ``top_k`` passages whether or not the answer needed them
    all, so reporting everything retrieved would attribute a water-damage
    answer to the personal-property clause. We keep the citations whose section
    the response names.

    If nothing matches but passages were retrieved, we fall back to all of
    them: a source that is hard to match is still better reported than
    silently dropped, and over-reporting is the safer error for a citation.
    """
    if not retrieved or not response:
        return retrieved

    used = [
        citation
        for citation in retrieved
        if citation in response or _section_of(citation) in response
    ]
    return used or retrieved


def _section_of(citation: str) -> str:
    """``file.md § Section 1: Title`` -> ``Section 1: Title``."""
    _, separator, section = citation.partition(" § ")
    return section if separator else citation

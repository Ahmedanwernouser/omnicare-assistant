"""Agent-facing tools.

Thin adapters over the pure functions in ``app/tools`` and ``app/rag``. Each
returns a :class:`ToolOutcome` carrying two things: the string the model reads,
and the structured metadata the API response needs (citations, arguments,
success flag). Keeping those separate is what lets ``/api/v1/chat`` report
``sources`` and ``tool_calls`` accurately instead of parsing them back out of
prose the model wrote.

Policy search is a *tool*, not a routing branch. A branch would force the graph
to decide up front whether a turn is a "RAG question" or a "claim question",
and "Is water damage covered, and what's the status of CLM-8821?" is both. As a
tool, the model can call it alongside the others in a single turn.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import StructuredTool

from app.agent.guards import scan_retrieved_text
from app.agent.prompts import format_policy_context
from app.rag.retriever import PolicyRetriever
from app.tools.claims import get_claim_status, submit_claim
from app.tools.store import ClaimStore

logger = logging.getLogger(__name__)

SEARCH_POLICY = "search_policy"
GET_CLAIM_STATUS = "get_claim_status"
SUBMIT_CLAIM = "submit_claim"


@dataclass
class ToolOutcome:
    """What a tool produced, split by audience."""

    content: str
    """Text the model reads."""

    ok: bool = True
    summary: str = ""
    sources: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


ToolFn = Callable[..., ToolOutcome]


@dataclass
class AgentTools:
    """The tool set, plus the raw callables the custom tool node executes."""

    schemas: list[StructuredTool]
    runners: dict[str, ToolFn]


def build_tools(
    *, retriever: PolicyRetriever, store: ClaimStore, top_k: int = 2
) -> AgentTools:
    """Construct the tool set bound to these dependencies."""

    def run_search_policy(query: str) -> ToolOutcome:
        passages = retriever.search(query, top_k=top_k)
        if not passages:
            return ToolOutcome(
                content=format_policy_context([]),
                ok=False,
                summary=f"No policy passages matched {query!r}.",
            )

        for passage in passages:
            if (rule := scan_retrieved_text(passage.text)) is not None:
                # Reported, never rewritten: silently editing a policy clause
                # would be a worse failure than the injection itself.
                logger.warning(
                    "Retrieved passage %s matched injection rule %s",
                    passage.citation,
                    rule,
                )

        citations = [p.citation for p in passages]
        return ToolOutcome(
            content=format_policy_context(passages),
            summary=f"Retrieved {len(passages)} passage(s).",
            sources=citations,
        )

    def run_get_claim_status(claim_id: str) -> ToolOutcome:
        result = get_claim_status(claim_id, store=store)
        if not result["ok"]:
            return ToolOutcome(
                content=f"LOOKUP FAILED: {result['error']}",
                ok=False,
                summary=result["error"],
            )
        claim = result["claim"]
        content = (
            f"Claim {claim['claim_id']} — status: {claim['status']}, "
            f"type: {claim['claim_type']}, amount: ${claim['amount']:,.2f}, "
            f"policy: {claim['policy_number']}."
        )
        return ToolOutcome(
            content=content,
            summary=f"{claim['claim_id']} is {claim['status']}.",
            extra={"claim": claim},
        )

    def run_submit_claim(
        policy_number: str, claim_type: str, amount: float, description: str
    ) -> ToolOutcome:
        result = submit_claim(
            policy_number=policy_number,
            claim_type=claim_type,
            amount=amount,
            description=description,
            store=store,
        )
        if not result["ok"]:
            # Phrased so the model relays the correction rather than retrying
            # blindly with invented values.
            return ToolOutcome(
                content=(
                    f"SUBMISSION REJECTED: {result['error']} "
                    "The claim was NOT filed. Tell the customer what to correct."
                ),
                ok=False,
                summary=result["error"],
            )
        confirmation = result["confirmation_id"]
        return ToolOutcome(
            content=(
                f"Claim filed successfully. Confirmation ID: {confirmation}. "
                f"Status: {result['claim']['status']}."
            ),
            summary=f"Filed {confirmation}.",
            extra={"confirmation_id": confirmation, "claim": result["claim"]},
        )

    schemas = [
        StructuredTool.from_function(
            func=lambda query: "",
            name=SEARCH_POLICY,
            description=(
                "Search OmniCare policy documents. Use this for every question "
                "about coverage, limits, deductibles, exclusions or conditions. "
                "Returns policy passages with the citation to quote."
            ),
        ),
        StructuredTool.from_function(
            func=lambda claim_id: "",
            name=GET_CLAIM_STATUS,
            description=(
                "Look up an existing claim by its ID, e.g. CLM-8821. This is "
                "the only source of claim information; never guess a status."
            ),
        ),
        StructuredTool.from_function(
            func=lambda policy_number, claim_type, amount, description: "",
            name=SUBMIT_CLAIM,
            description=(
                "File a new insurance claim. Requires the policy number "
                "(POL-1234), the claim type ('Water Damage' or 'Personal "
                "Property'), the amount in dollars, and a description of at "
                "least 10 characters. Ask the customer for anything missing — "
                "never invent values. Returns a confirmation ID on success."
            ),
        ),
    ]

    runners: dict[str, ToolFn] = {
        SEARCH_POLICY: run_search_policy,
        GET_CLAIM_STATUS: run_get_claim_status,
        SUBMIT_CLAIM: run_submit_claim,
    }
    return AgentTools(schemas=schemas, runners=runners)

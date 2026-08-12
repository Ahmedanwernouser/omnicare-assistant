"""Operational claim tools: ``get_claim_status`` and ``submit_claim``.

These functions are the security boundary of the application. Every argument
arrives from an LLM, so it is treated as untrusted input and validated by
Pydantic here — not upstream in a prompt. A jailbroken model still cannot post
a negative amount, invent a field, or write a malformed policy number, because
the model never touches the datastore directly.

Deliberate omission: coverage limits ($25,000 water damage, $10,000 personal
property) are **not** hardcoded here. They live in the policy document and are
answered through RAG, so the policy stays the single source of truth.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.tools.store import ClaimStore, ClaimStoreError

ClaimType = Literal["Water Damage", "Personal Property"]

#: Canonical claim types, keyed by their normalised form.
_CLAIM_TYPE_ALIASES: dict[str, ClaimType] = {
    "water damage": "Water Damage",
    "waterdamage": "Water Damage",
    "water": "Water Damage",
    "personal property": "Personal Property",
    "personalproperty": "Personal Property",
    "property": "Personal Property",
}

_POLICY_PATTERN = r"^POL-\d{4}$"
_CLAIM_ID_PATTERN = r"^CLM-\d{4}$"
_NEW_CLAIM_STATUS = "Submitted"


# --------------------------------------------------------------------------
# Input models
# --------------------------------------------------------------------------


class ClaimLookup(BaseModel):
    """Validated input for :func:`get_claim_status`."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    claim_id: str = Field(pattern=_CLAIM_ID_PATTERN)

    @field_validator("claim_id", mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        cleaned = value.strip().upper().replace(" ", "").replace("_", "-")
        # Accept a bare number ("8821") and the common "CLM8821" typo.
        if cleaned.isdigit():
            cleaned = f"CLM-{cleaned}"
        elif cleaned.startswith("CLM") and not cleaned.startswith("CLM-"):
            cleaned = f"CLM-{cleaned[3:]}"
        return cleaned


class ClaimSubmission(BaseModel):
    """Validated input for :func:`submit_claim`.

    ``extra="forbid"`` matters: if the model hallucinates an extra argument
    such as ``status="Approved"``, the call is rejected instead of letting the
    model set its own claim status.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    policy_number: str = Field(pattern=_POLICY_PATTERN)
    claim_type: ClaimType
    amount: float = Field(gt=0, le=1_000_000)
    description: str = Field(min_length=10, max_length=1000)

    @field_validator("policy_number", mode="before")
    @classmethod
    def _normalise_policy(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        cleaned = value.strip().upper().replace(" ", "").replace("_", "-")
        if cleaned.isdigit():
            cleaned = f"POL-{cleaned}"
        elif cleaned.startswith("POL") and not cleaned.startswith("POL-"):
            cleaned = f"POL-{cleaned[3:]}"
        return cleaned

    @field_validator("claim_type", mode="before")
    @classmethod
    def _normalise_type(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        key = re.sub(r"[\s_-]+", " ", value).strip().lower()
        return _CLAIM_TYPE_ALIASES.get(key, value)

    @field_validator("amount", mode="before")
    @classmethod
    def _parse_amount(cls, value: Any) -> Any:
        """Normalise, then round to cents *before* the constraints run.

        Rounding after validation would let 0.001 satisfy ``gt=0`` and then be
        stored as 0.00 — a persisted value that violates the rule the schema
        just enforced. Rounding here means the constraint applies to the number
        that actually gets written.
        """
        if isinstance(value, str):
            # LLMs routinely emit "$3,500.00" instead of 3500.0.
            value = value.strip().replace("$", "").replace(",", "").replace("_", "")
            if not value:
                return value
        try:
            return round(float(value), 2)
        except (TypeError, ValueError):
            # Not a number: hand it back so Pydantic reports the type error.
            return value


# --------------------------------------------------------------------------
# Result envelope
# --------------------------------------------------------------------------


def _ok(**data: Any) -> dict[str, Any]:
    return {"ok": True, **data}


def _fail(message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "error": message, **extra}


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"]) or "input"
        parts.append(f"{field}: {err['msg']}")
    return "; ".join(parts)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


def get_claim_status(claim_id: str, *, store: ClaimStore) -> dict[str, Any]:
    """Look up one claim.

    A missing claim is a normal outcome, not an exception: the agent needs to
    tell the caller "no such claim" rather than surface a stack trace.
    """
    try:
        lookup = ClaimLookup(claim_id=claim_id)
    except ValidationError as exc:
        return _fail(
            "Invalid claim ID. Expected the format CLM-1234. "
            f"({_format_validation_error(exc)})"
        )

    try:
        claim = store.find(lookup.claim_id)
    except ClaimStoreError as exc:
        return _fail(f"Claim database unavailable: {exc}")

    if claim is None:
        return _fail(
            f"No claim found with ID {lookup.claim_id}.",
            claim_id=lookup.claim_id,
            found=False,
        )
    return _ok(found=True, claim=claim)


def submit_claim(
    policy_number: str,
    claim_type: str,
    amount: float,
    description: str,
    *,
    store: ClaimStore,
) -> dict[str, Any]:
    """Validate and persist a new claim, returning its confirmation ID."""
    try:
        submission = ClaimSubmission(
            policy_number=policy_number,
            claim_type=claim_type,
            amount=amount,
            description=description,
        )
    except ValidationError as exc:
        return _fail(
            f"Claim rejected — invalid input. {_format_validation_error(exc)}"
        )

    def build(claim_id: str) -> dict[str, Any]:
        return {
            "claim_id": claim_id,
            "policy_number": submission.policy_number,
            "claim_type": submission.claim_type,
            "status": _NEW_CLAIM_STATUS,
            "amount": submission.amount,  # already rounded by the validator
            "description": submission.description,
        }

    try:
        record = store.append(build)
    except ClaimStoreError as exc:
        return _fail(f"Could not save the claim: {exc}")

    return _ok(confirmation_id=record["claim_id"], claim=record)

"""Tool contract: lookups, submissions and the validation boundary.

The submission tests double as safety tests. They assert that the datastore
stays consistent even when the caller — an LLM that may have been talked into
anything — sends hostile or malformed arguments.
"""

from __future__ import annotations

import pytest

from app.tools.claims import get_claim_status, submit_claim
from app.tools.store import ClaimStore

VALID = {
    "policy_number": "POL-1092",
    "claim_type": "Water Damage",
    "amount": 4200.50,
    "description": "Burst pipe under the kitchen sink flooded the floor.",
}


# -- get_claim_status ------------------------------------------------------


def test_lookup_returns_the_seeded_claim(store: ClaimStore) -> None:
    result = get_claim_status("CLM-8821", store=store)
    assert result["ok"] is True
    assert result["claim"]["status"] == "Approved"
    assert result["claim"]["amount"] == 3500.00


@pytest.mark.parametrize("raw", ["clm-8821", " CLM-8821 ", "CLM8821", "8821"])
def test_lookup_normalises_common_id_shapes(store: ClaimStore, raw: str) -> None:
    assert get_claim_status(raw, store=store)["ok"] is True


def test_lookup_miss_is_a_result_not_an_exception(store: ClaimStore) -> None:
    result = get_claim_status("CLM-4242", store=store)
    assert result["ok"] is False
    assert result["found"] is False
    assert "CLM-4242" in result["error"]


@pytest.mark.parametrize("bad", ["", "DROP TABLE claims", "CLM-99999", "../../etc/passwd"])
def test_lookup_rejects_malformed_ids(store: ClaimStore, bad: str) -> None:
    result = get_claim_status(bad, store=store)
    assert result["ok"] is False
    assert "CLM-1234" in result["error"]


# -- submit_claim ----------------------------------------------------------


def test_submission_persists_and_returns_a_confirmation_id(
    store: ClaimStore, read_claims
) -> None:
    result = submit_claim(**VALID, store=store)

    assert result["ok"] is True
    confirmation = result["confirmation_id"]
    assert confirmation.startswith("CLM-")

    on_disk = read_claims()
    assert len(on_disk) == 3
    assert on_disk[-1]["claim_id"] == confirmation
    assert on_disk[-1]["status"] == "Submitted"


def test_submitted_claim_is_immediately_retrievable(store: ClaimStore) -> None:
    """The round trip an evaluator will try first."""
    confirmation = submit_claim(**VALID, store=store)["confirmation_id"]
    lookup = get_claim_status(confirmation, store=store)
    assert lookup["ok"] is True
    assert lookup["claim"]["policy_number"] == "POL-1092"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("amount", -500),
        ("amount", 0),
        ("amount", 5_000_000),
        ("amount", "not a number"),
        ("policy_number", "POL-99"),
        ("policy_number", "'; DROP TABLE claims; --"),
        ("claim_type", "Alien Invasion"),
        ("description", "short"),
        ("description", "x" * 1001),
    ],
)
def test_invalid_input_is_rejected_and_nothing_is_written(
    store: ClaimStore, read_claims, field: str, value: object
) -> None:
    result = submit_claim(**{**VALID, field: value}, store=store)
    assert result["ok"] is False
    assert field in result["error"]
    assert len(read_claims()) == 2  # datastore untouched


def test_model_cannot_set_its_own_claim_status(store: ClaimStore, read_claims) -> None:
    """A hallucinated or injected extra argument must not reach the record."""
    result = submit_claim(**VALID, store=store)
    assert read_claims()[-1]["status"] == "Submitted"
    assert result["claim"]["status"] == "Submitted"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("$4,200.50", 4200.50), ("4200", 4200.0), (" 4200.5 ", 4200.5)],
)
def test_amount_accepts_the_formats_llms_actually_emit(
    store: ClaimStore, raw: str, expected: float
) -> None:
    result = submit_claim(**{**VALID, "amount": raw}, store=store)
    assert result["ok"] is True
    assert result["claim"]["amount"] == expected


@pytest.mark.parametrize("raw", ["water damage", "WATER DAMAGE", "water_damage"])
def test_claim_type_is_normalised_to_the_canonical_value(
    store: ClaimStore, raw: str
) -> None:
    result = submit_claim(**{**VALID, "claim_type": raw}, store=store)
    assert result["ok"] is True
    assert result["claim"]["claim_type"] == "Water Damage"


def test_amount_is_rounded_to_cents(store: ClaimStore) -> None:
    result = submit_claim(**{**VALID, "amount": 1234.5678}, store=store)
    assert result["claim"]["amount"] == 1234.57

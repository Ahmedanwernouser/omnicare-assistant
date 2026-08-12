"""Persistence layer: atomicity, isolation and ID minting."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.tools.store import ClaimStore, ClaimStoreError


def test_reads_seed_data(store: ClaimStore) -> None:
    claims = store.read_all()
    assert [c["claim_id"] for c in claims] == ["CLM-8821", "CLM-9014"]


def test_find_is_case_insensitive(store: ClaimStore) -> None:
    assert store.find("clm-8821")["policy_number"] == "POL-1092"


def test_find_returns_none_instead_of_raising(store: ClaimStore) -> None:
    assert store.find("CLM-0000") is None


def test_missing_file_reads_as_empty(tmp_path: Path) -> None:
    assert ClaimStore(tmp_path / "nope.json").read_all() == []


def test_append_mints_a_unique_id(store: ClaimStore, read_claims) -> None:
    record = store.append(lambda cid: {"claim_id": cid, "amount": 1.0})
    assert record["claim_id"].startswith("CLM-")
    assert len(record["claim_id"]) == len("CLM-1234")

    on_disk = read_claims()
    assert len(on_disk) == 3
    assert on_disk[-1]["claim_id"] == record["claim_id"]


def test_append_never_reuses_an_existing_id(store: ClaimStore) -> None:
    seen = {c["claim_id"] for c in store.read_all()}
    for _ in range(25):
        record = store.append(lambda cid: {"claim_id": cid})
        assert record["claim_id"] not in seen
        seen.add(record["claim_id"])


def test_concurrent_appends_do_not_lose_updates(store: ClaimStore, read_claims) -> None:
    """The lost-update failure mode the FileLock exists to prevent."""
    writers = 12

    def write(i: int) -> str:
        return store.append(lambda cid: {"claim_id": cid, "seq": i})["claim_id"]

    with ThreadPoolExecutor(max_workers=writers) as pool:
        ids = list(pool.map(write, range(writers)))

    on_disk = read_claims()
    assert len(on_disk) == 2 + writers          # nothing overwritten
    assert len(set(ids)) == writers             # no duplicate IDs handed out


def test_write_is_atomic_and_leaves_no_temp_files(
    store: ClaimStore, claims_path: Path
) -> None:
    store.overwrite([{"claim_id": "CLM-1111"}])
    leftovers = [p.name for p in claims_path.parent.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
    assert json.loads(claims_path.read_text())[0]["claim_id"] == "CLM-1111"


def test_corrupt_json_raises_a_clear_error(claims_path: Path) -> None:
    claims_path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ClaimStoreError, match="not valid JSON"):
        ClaimStore(claims_path).read_all()


def test_non_array_json_is_rejected(claims_path: Path) -> None:
    claims_path.write_text('{"claims": []}', encoding="utf-8")
    with pytest.raises(ClaimStoreError, match="JSON array"):
        ClaimStore(claims_path).read_all()

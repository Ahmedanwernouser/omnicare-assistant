"""Persistence layer for the mock claims database.

The assessment specifies a JSON file as the datastore. A JSON file has two
failure modes that a real database would handle for us:

1. **Lost updates.** Two concurrent ``submit_claim`` calls both read the list,
   both append, and the second write silently discards the first claim.
2. **Torn writes.** A crash midway through ``json.dump`` leaves a truncated
   file, so the database is destroyed rather than merely stale.

We close both: an inter-process ``FileLock`` serialises read-modify-write
cycles, and every write goes to a temporary file in the same directory which is
then moved into place with ``os.replace`` (atomic on POSIX and Windows).

Claim IDs are generated *inside* the lock, immediately before the append, so
two callers can never be handed the same confirmation ID.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

Claim = dict[str, Any]

_ID_PREFIX = "CLM-"
_ID_DIGITS = 4
_ID_MAX_ATTEMPTS = 50


class ClaimStoreError(RuntimeError):
    """Raised when the store cannot be read or written safely."""


class ClaimStore:
    """Concurrency-safe accessor for a JSON array of claim records."""

    def __init__(self, path: Path | str, lock_timeout: float = 10.0) -> None:
        self._path = Path(path)
        self._lock = FileLock(f"{self._path}.lock", timeout=lock_timeout)

    @property
    def path(self) -> Path:
        return self._path

    # -- public API --------------------------------------------------------

    def read_all(self) -> list[Claim]:
        """Return every claim record. Missing file is treated as empty."""
        with self._acquire():
            return self._read_unlocked()

    def find(self, claim_id: str) -> Claim | None:
        """Return the claim with ``claim_id`` or ``None``. Never raises on miss."""
        target = claim_id.strip().upper()
        with self._acquire():
            for claim in self._read_unlocked():
                if str(claim.get("claim_id", "")).upper() == target:
                    return dict(claim)
        return None

    def append(self, build_record: Callable[[str], Claim]) -> Claim:
        """Append one record, atomically.

        ``build_record`` receives a freshly generated, collision-checked claim
        ID and returns the full record to persist. Passing a callback rather
        than a finished dict is what lets us mint the ID while still holding
        the lock.
        """
        with self._acquire():
            claims = self._read_unlocked()
            taken = {str(c.get("claim_id", "")).upper() for c in claims}
            record = build_record(self._mint_id(taken))
            claims.append(record)
            self._write_unlocked(claims)
            return dict(record)

    def overwrite(self, claims: list[Claim]) -> None:
        """Replace the whole file. Used by tests and the seed routine."""
        with self._acquire():
            self._write_unlocked(claims)

    # -- internals ---------------------------------------------------------

    def _acquire(self) -> FileLock:
        try:
            return self._lock.acquire()
        except Timeout as exc:  # pragma: no cover - contention is rare in tests
            raise ClaimStoreError(
                f"Timed out waiting for a lock on {self._path}."
            ) from exc

    def _read_unlocked(self) -> list[Claim]:
        if not self._path.exists():
            return []
        try:
            # utf-8-sig strips a BOM if present. A Windows editor adding one
            # would otherwise make the whole claims file unreadable.
            raw = self._path.read_text(encoding="utf-8-sig").strip()
        except OSError as exc:
            raise ClaimStoreError(f"Cannot read {self._path}: {exc}") from exc
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ClaimStoreError(
                f"{self._path} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(data, list):
            raise ClaimStoreError(
                f"{self._path} must contain a JSON array, got {type(data).__name__}."
            )
        return data

    def _write_unlocked(self, claims: list[Claim]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name: str | None = None
        try:
            # Same directory as the target, so os.replace stays on one filesystem.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_name = tmp.name
                json.dump(claims, tmp, indent=2, ensure_ascii=False)
                tmp.write("\n")
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_name, self._path)
            tmp_name = None
        except OSError as exc:
            raise ClaimStoreError(f"Cannot write {self._path}: {exc}") from exc
        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)

    @staticmethod
    def _mint_id(taken: set[str]) -> str:
        lo, hi = 10 ** (_ID_DIGITS - 1), 10**_ID_DIGITS - 1
        for _ in range(_ID_MAX_ATTEMPTS):
            candidate = f"{_ID_PREFIX}{random.randint(lo, hi)}"
            if candidate not in taken:
                return candidate
        raise ClaimStoreError(
            "Exhausted the claim ID space; the mock database is full."
        )

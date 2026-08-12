#!/usr/bin/env python3
"""Download and cache the embedding model.

Run at Docker **build** time, never at request time. Chroma fetches
all-MiniLM-L6-v2 (~80 MB) from S3 the first time it embeds anything; leaving
that to the first user request means the demo either stalls for a minute or
fails outright behind a restrictive network or proxy.

    python scripts/warm_embeddings.py

Exits non-zero on failure so a broken image fails the build rather than
failing in front of a reviewer.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.embeddings import build_embedding_function  # noqa: E402


def main() -> int:
    started = time.perf_counter()
    print("Warming embedding model (all-MiniLM-L6-v2)...", flush=True)
    try:
        embed = build_embedding_function("onnx")
        vectors = embed(["warm up the ONNX session and the tokenizer"])
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "\nThe model is fetched from chroma-onnx-models.s3.amazonaws.com. "
            "Check egress from the build environment.",
            file=sys.stderr,
        )
        return 1

    elapsed = time.perf_counter() - started
    print(f"OK - {len(vectors[0])}-dim vectors, cached in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download and cache the embedding model.

Run at Docker **build** time, never at request time. Chroma fetches
all-MiniLM-L6-v2 (~80 MB) from S3 the first time it embeds anything; leaving
that to the first user request means the demo either stalls for a minute or
fails outright behind a restrictive network or proxy.

    python scripts/warm_embeddings.py

Two things this script deliberately does *not* do:

* **It does not import the application.** Depending only on ``chromadb`` lets
  the Dockerfile warm the model straight after ``pip install``, before any
  source is copied — so editing a Python file does not invalidate the layer
  and trigger another 80 MB download.
* **It does not guess where the cache goes.** Chroma writes to
  ``Path.home()/.cache/chroma``, which follows ``$HOME``. Warming as root and
  running as a different user silently misses the cache, so the Dockerfile
  switches to the runtime user *before* calling this.

Exits non-zero on failure, so a broken image fails the build rather than
failing in front of a reviewer.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> int:
    started = time.perf_counter()
    home = Path.home()
    print(f"Warming all-MiniLM-L6-v2 into {home}/.cache/chroma ...", flush=True)

    try:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        vectors = ONNXMiniLM_L6_V2()(["warm up the ONNX session and tokenizer"])
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "\nThe model is fetched from chroma-onnx-models.s3.amazonaws.com. "
            "Check egress from the build environment, then rebuild.",
            file=sys.stderr,
        )
        return 1

    cached = home / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"
    print(
        f"OK - {len(vectors[0])}-dim vectors in {time.perf_counter() - started:.1f}s\n"
        f"Cached at {cached} (exists: {cached.exists()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

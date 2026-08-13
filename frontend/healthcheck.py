#!/usr/bin/env python3
"""Container health probe.

A script rather than an inline `python -c` in the Dockerfile: the inline form
needs a JSON array with an escaped line continuation, which is easy to get
subtly wrong and impossible to test outside a build. This can be run directly.

Streamlit exposes /_stcore/health once the server is accepting connections.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

URL = os.getenv("HEALTHCHECK_URL", "http://127.0.0.1:8501/_stcore/health")
TIMEOUT = float(os.getenv("HEALTHCHECK_TIMEOUT", "4"))


def main() -> int:
    try:
        with urllib.request.urlopen(URL, timeout=TIMEOUT) as response:
            return 0 if response.status == 200 else 1
    except urllib.error.HTTPError as exc:
        # 503 while the index loads is expected, not a crash.
        print(f"not ready: HTTP {exc.code}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"not ready: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

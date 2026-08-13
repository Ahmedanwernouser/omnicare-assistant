#!/usr/bin/env python3
"""Print the agent graph, straight from the compiled StateGraph.

    python scripts/draw_graph.py

Mermaid only. ASCII output would pull in grandalf for pure decoration, and
GitHub renders mermaid natively — which is where the README is read.

The README's generated diagram comes from here, so documentation cannot drift
from behaviour: if a node or edge changes, this output changes with it.

Uses the scripted model and the lexical embedding backend, so it needs no API
key and no network.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from langchain_core.messages import AIMessage  # noqa: E402

from app.agent.graph import PolicyAgent  # noqa: E402
from app.agent.llm import ScriptedChatModel  # noqa: E402
from app.agent.tools import build_tools  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.rag.retriever import PolicyRetriever  # noqa: E402
from app.tools.store import ClaimStore  # noqa: E402


def build_agent(workdir: Path) -> PolicyAgent:
    settings = get_settings()
    shutil.copytree(settings.data_dir, workdir / "data")
    retriever = PolicyRetriever(
        persist_dir=workdir / "chroma",
        collection_name="diagram_only",
        embedding_backend="lexical",
    )
    return PolicyAgent(
        chat_model=ScriptedChatModel(responses=[AIMessage(content="")]),
        tools=build_tools(
            retriever=retriever,
            store=ClaimStore(workdir / "data" / "mock_claims.json"),
        ),
    )


def main() -> int:
    workdir = Path(tempfile.mkdtemp())
    try:
        graph = build_agent(workdir).graph.get_graph()
        print(graph.draw_mermaid())
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

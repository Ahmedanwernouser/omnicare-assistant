"""Split a policy markdown document into citable chunks.

Chunking strategy is driven by the citation requirement, not by a token
budget. The assessment asks for citations in the answer, and a citation is only
useful to a policyholder if it names the clause they can go read — so the
``##`` heading is the unit we split on, and every chunk carries its heading
through to the answer as ``sample_policy.md § Section 1: ...``.

Two details that are easy to get wrong:

* **Contextual headers.** The text sent to the embedding model is prefixed with
  the document title and section heading. A question about "burst pipes" has to
  match a passage whose own words are "sudden pipe bursts"; carrying
  "Home Water Damage Coverage" into the embedded text gives the retriever the
  vocabulary bridge it needs.
* **Oversized sections.** The provided document has tiny sections, but a real
  policy does not. Sections longer than ``max_chunk_chars`` are split again on
  paragraph and then sentence boundaries, and the parts stay individually
  citable as ``... (part 2/3)``.

This module has no third-party dependencies on purpose: it is the part of the
RAG pipeline that can be tested exhaustively without a vector store or a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MAX_CHUNK_CHARS = 1200
_UNTITLED = "Preamble"

_H1 = re.compile(r"^#\s+(?P<title>.+?)\s*$")
_H2 = re.compile(r"^##\s+(?P<title>.+?)\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class PolicyChunk:
    """One citable passage of a policy document."""

    text: str
    """Raw passage text, as it appears in the source document."""

    source_file: str
    section_title: str
    doc_title: str = ""
    part: int = 1
    total_parts: int = 1
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """Human-readable citation, e.g.
        ``sample_policy.md § Section 1: Home Water Damage Coverage``."""
        base = f"{self.source_file} § {self.section_title}"
        if self.total_parts > 1:
            return f"{base} (part {self.part}/{self.total_parts})"
        return base

    @property
    def chunk_id(self) -> str:
        """Stable ID, so re-ingesting the same document upserts rather than
        duplicates."""
        slug = re.sub(r"[^a-z0-9]+", "-", self.section_title.lower()).strip("-")
        return f"{self.source_file}::{slug}::{self.part}"

    @property
    def embedding_text(self) -> str:
        """Text handed to the embedding model, with its heading context."""
        header = " > ".join(p for p in (self.doc_title, self.section_title) if p)
        return f"{header}\n\n{self.text}" if header else self.text

    def as_chroma_metadata(self) -> dict[str, str | int]:
        return {
            "source_file": self.source_file,
            "section_title": self.section_title,
            "doc_title": self.doc_title,
            "part": self.part,
            "total_parts": self.total_parts,
            "citation": self.citation,
            **self.metadata,
        }


def split_policy_markdown(
    markdown: str,
    *,
    source_file: str,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[PolicyChunk]:
    """Split ``markdown`` into citable chunks, one per ``##`` section."""
    doc_title = ""
    section_title = _UNTITLED
    buffer: list[str] = []
    sections: list[tuple[str, str]] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((section_title, body))
        buffer.clear()

    for line in markdown.splitlines():
        if (m := _H2.match(line)) is not None:
            flush()
            section_title = m.group("title")
            continue
        if (m := _H1.match(line)) is not None and not doc_title:
            flush()
            doc_title = m.group("title")
            section_title = _UNTITLED
            continue
        buffer.append(line)
    flush()

    chunks: list[PolicyChunk] = []
    for title, body in sections:
        parts = _split_oversized(body, max_chunk_chars)
        total = len(parts)
        for index, part_text in enumerate(parts, start=1):
            chunks.append(
                PolicyChunk(
                    text=part_text,
                    source_file=source_file,
                    section_title=title,
                    doc_title=doc_title,
                    part=index,
                    total_parts=total,
                )
            )
    return chunks


def load_policy_chunks(
    path: Path | str, *, max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS
) -> list[PolicyChunk]:
    """Read a policy document from disk and split it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Policy document not found: {path}")
    return split_policy_markdown(
        path.read_text(encoding="utf-8"),
        source_file=path.name,
        max_chunk_chars=max_chunk_chars,
    )


# --------------------------------------------------------------------------
# Oversized-section handling
# --------------------------------------------------------------------------


def _split_oversized(body: str, limit: int) -> list[str]:
    if len(body) <= limit:
        return [body]

    pieces: list[str] = []
    current = ""
    for unit in _units(body, limit):
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if current and len(candidate) > limit:
            pieces.append(current)
            current = unit
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces or [body]


def _units(body: str, limit: int) -> list[str]:
    """Paragraphs, further broken into sentences when a paragraph alone is
    still over the limit."""
    units: list[str] = []
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", body)):
        if not paragraph:
            continue
        if len(paragraph) <= limit:
            units.append(paragraph)
            continue
        units.extend(_hard_wrap(s for s in _SENTENCE_END.split(paragraph) if s))
    return units


def _hard_wrap(sentences) -> list[str]:
    """Last resort for a single sentence longer than the limit: keep it whole
    rather than cutting mid-word. Truncating a policy clause would be worse
    than an oversized chunk."""
    return [s.strip() for s in sentences if s.strip()]

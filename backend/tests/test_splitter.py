"""Splitter behaviour. No vector store, no model, no network."""

from __future__ import annotations

import pytest

from app.rag.splitter import (
    PolicyChunk,
    load_policy_chunks,
    split_policy_markdown,
)

SAMPLE = """# OmniCare General Insurance Policy 2026

## Section 1: Home Water Damage Coverage

Water damage caused by sudden pipe bursts is covered up to $25,000 with a
$500 deductible. Gradual leaks or flood damage are strictly excluded.

## Section 2: Personal Property Protection

Electronics, furniture, and jewelry are covered up to $10,000 total.
"""


@pytest.fixture
def chunks() -> list[PolicyChunk]:
    return split_policy_markdown(SAMPLE, source_file="sample_policy.md")


# -- structure -------------------------------------------------------------


def test_one_chunk_per_h2_section(chunks: list[PolicyChunk]) -> None:
    assert len(chunks) == 2
    assert [c.section_title for c in chunks] == [
        "Section 1: Home Water Damage Coverage",
        "Section 2: Personal Property Protection",
    ]


def test_h1_becomes_the_document_title_not_a_section(chunks: list[PolicyChunk]) -> None:
    assert all(c.doc_title == "OmniCare General Insurance Policy 2026" for c in chunks)
    assert not any("2026" in c.section_title for c in chunks)


def test_section_body_is_preserved_verbatim(chunks: list[PolicyChunk]) -> None:
    assert "$25,000" in chunks[0].text
    assert "$500 deductible" in chunks[0].text
    assert "strictly excluded" in chunks[0].text
    assert "##" not in chunks[0].text


def test_sections_do_not_bleed_into_each_other(chunks: list[PolicyChunk]) -> None:
    assert "jewelry" not in chunks[0].text
    assert "pipe bursts" not in chunks[1].text


# -- citations -------------------------------------------------------------


def test_citation_names_the_clause_not_just_the_file(chunks: list[PolicyChunk]) -> None:
    assert (
        chunks[0].citation
        == "sample_policy.md § Section 1: Home Water Damage Coverage"
    )


def test_citations_are_unique(chunks: list[PolicyChunk]) -> None:
    assert len({c.citation for c in chunks}) == len(chunks)


def test_chunk_ids_are_stable_across_reingestion() -> None:
    a = split_policy_markdown(SAMPLE, source_file="sample_policy.md")
    b = split_policy_markdown(SAMPLE, source_file="sample_policy.md")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


def test_embedding_text_carries_heading_context(chunks: list[PolicyChunk]) -> None:
    """The vocabulary bridge from 'burst pipe' to 'Water Damage Coverage'."""
    assert "Water Damage Coverage" in chunks[0].embedding_text
    assert "pipe bursts" in chunks[0].embedding_text


def test_metadata_round_trips_for_chroma(chunks: list[PolicyChunk]) -> None:
    meta = chunks[0].as_chroma_metadata()
    assert meta["citation"] == chunks[0].citation
    assert meta["section_title"] == "Section 1: Home Water Damage Coverage"
    # Chroma only accepts scalar metadata values.
    assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())


# -- edge cases ------------------------------------------------------------


def test_text_before_the_first_heading_is_kept() -> None:
    out = split_policy_markdown(
        "Intro sentence with no heading.", source_file="x.md"
    )
    assert len(out) == 1
    assert out[0].section_title == "Preamble"


def test_empty_document_yields_no_chunks() -> None:
    assert split_policy_markdown("   \n\n  ", source_file="x.md") == []


def test_empty_sections_are_dropped() -> None:
    out = split_policy_markdown("## Empty\n\n## Real\n\nBody.", source_file="x.md")
    assert [c.section_title for c in out] == ["Real"]


def test_oversized_section_is_split_and_parts_stay_citable() -> None:
    body = "\n\n".join(f"Paragraph number {i} with filler text." * 6 for i in range(12))
    out = split_policy_markdown(
        f"# Doc\n\n## Big Section\n\n{body}", source_file="x.md", max_chunk_chars=400
    )
    assert len(out) > 1
    assert all(c.section_title == "Big Section" for c in out)
    assert out[0].citation == "x.md § Big Section (part 1/%d)" % len(out)
    assert len({c.chunk_id for c in out}) == len(out)


def test_a_single_overlong_clause_is_never_truncated() -> None:
    clause = "The insurer shall indemnify the policyholder " * 40
    out = split_policy_markdown(
        f"## Clause\n\n{clause}", source_file="x.md", max_chunk_chars=100
    )
    assert "".join(c.text for c in out).count("indemnify") == 40


# -- disk loading ----------------------------------------------------------


def test_loads_the_real_policy_document(data_dir) -> None:
    out = load_policy_chunks(data_dir / "sample_policy.md")
    assert len(out) == 2
    assert out[0].source_file == "sample_policy.md"
    assert "25,000" in out[0].text


def test_missing_file_raises_a_clear_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Policy document not found"):
        load_policy_chunks(tmp_path / "nope.md")

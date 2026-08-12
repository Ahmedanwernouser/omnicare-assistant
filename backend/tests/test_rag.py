"""Retrieval behaviour.

Split into two tiers on purpose:

* The default tier runs on the ``lexical`` backend — deterministic, offline,
  no model download — and proves the *plumbing*: ingestion, ranking, metadata,
  citation strings, idempotency, backend migration.
* ``test_real_model_understands_paraphrase`` runs on the real ONNX model and
  proves *semantics*: that "burst pipe" reaches a passage that never uses the
  word "burst" in the question's phrasing. It is skipped automatically when the
  model is not cached, so ``pytest`` still passes offline and keyless.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from app.rag.retriever import PolicyRetriever
from app.rag.splitter import load_policy_chunks

warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture
def retriever(tmp_path: Path, data_dir: Path) -> PolicyRetriever:
    r = PolicyRetriever(
        persist_dir=tmp_path / "chroma",
        collection_name="test_policy",
        embedding_backend="lexical",
        top_k=2,
    )
    r.ingest_file(data_dir / "sample_policy.md")
    return r


# -- ingestion -------------------------------------------------------------


def test_ingests_one_vector_per_section(retriever: PolicyRetriever) -> None:
    assert retriever.count() == 2


def test_reingestion_upserts_instead_of_duplicating(
    retriever: PolicyRetriever, data_dir: Path
) -> None:
    """Restarting the API must not grow the collection."""
    for _ in range(3):
        retriever.ingest_file(data_dir / "sample_policy.md")
    assert retriever.count() == 2


def test_ingesting_nothing_is_a_no_op(retriever: PolicyRetriever) -> None:
    assert retriever.ingest([]) == 0
    assert retriever.count() == 2


# -- ranking ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected_section"),
    [
        ("burst pipe water damage", "Section 1: Home Water Damage Coverage"),
        ("deductible", "Section 1: Home Water Damage Coverage"),
        ("jewelry electronics furniture", "Section 2: Personal Property Protection"),
        ("appraisal receipts", "Section 2: Personal Property Protection"),
    ],
)
def test_top_hit_is_the_right_clause(
    retriever: PolicyRetriever, query: str, expected_section: str
) -> None:
    hits = retriever.search(query)
    assert hits, f"no passages returned for {query!r}"
    assert hits[0].section_title == expected_section


def test_results_are_ordered_by_closeness(retriever: PolicyRetriever) -> None:
    hits = retriever.search("burst pipe water damage")
    assert [h.distance for h in hits] == sorted(h.distance for h in hits)


def test_top_k_is_respected(retriever: PolicyRetriever) -> None:
    assert len(retriever.search("coverage", top_k=1)) == 1


def test_top_k_larger_than_the_corpus_does_not_error(
    retriever: PolicyRetriever,
) -> None:
    assert len(retriever.search("coverage", top_k=50)) == 2


# -- citations -------------------------------------------------------------


def test_every_passage_carries_a_section_level_citation(
    retriever: PolicyRetriever,
) -> None:
    for hit in retriever.search("coverage"):
        assert hit.citation.startswith("sample_policy.md § Section ")
        assert hit.section_title in hit.citation


def test_passage_text_is_the_clause_not_the_embedding_wrapper(
    retriever: PolicyRetriever,
) -> None:
    """The answer must quote the policy, not our synthetic heading prefix."""
    hit = retriever.search("burst pipe")[0]
    assert "$25,000" in hit.text
    assert not hit.text.startswith("OmniCare General Insurance Policy")


def test_relevance_is_derived_from_distance(retriever: PolicyRetriever) -> None:
    hit = retriever.search("burst pipe")[0]
    assert hit.relevance == pytest.approx(1.0 - hit.distance, abs=1e-4)


# -- edge cases ------------------------------------------------------------


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_returns_nothing(retriever: PolicyRetriever, query: str) -> None:
    assert retriever.search(query) == []


def test_search_on_an_empty_collection_returns_nothing(tmp_path: Path) -> None:
    empty = PolicyRetriever(
        persist_dir=tmp_path / "chroma",
        collection_name="empty",
        embedding_backend="lexical",
    )
    assert empty.search("anything") == []


def test_reset_clears_the_collection(retriever: PolicyRetriever) -> None:
    retriever.reset()
    assert retriever.count() == 0


# -- persistence & migration ----------------------------------------------


def test_collection_survives_a_restart(tmp_path: Path, data_dir: Path) -> None:
    persist = tmp_path / "chroma"
    PolicyRetriever(
        persist_dir=persist, collection_name="restart_test", embedding_backend="lexical"
    ).ingest_file(data_dir / "sample_policy.md")

    reopened = PolicyRetriever(
        persist_dir=persist, collection_name="restart_test", embedding_backend="lexical"
    )
    assert reopened.count() == 2
    assert reopened.search("burst pipe")[0].citation.endswith(
        "Section 1: Home Water Damage Coverage"
    )


def test_switching_embedding_backend_rebuilds_instead_of_crashing(
    tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Different backends produce different vector dimensions, so reopening a
    collection under a new backend must rebuild rather than fail at query time.

    The second backend is stubbed to the lexical function so the migration path
    itself is exercised offline — what is under test is the rebuild decision,
    not the model.
    """
    from app.rag import retriever as retriever_module
    from app.rag.embeddings import LexicalEmbeddingFunction

    persist = tmp_path / "chroma"
    first = PolicyRetriever(
        persist_dir=persist,
        collection_name="migration_test",
        embedding_backend="lexical",
    )
    first.ingest_file(data_dir / "sample_policy.md")
    assert first.count() == 2

    monkeypatch.setattr(
        retriever_module,
        "build_embedding_function",
        lambda backend: LexicalEmbeddingFunction(),
    )
    second = PolicyRetriever(
        persist_dir=persist,
        collection_name="migration_test",
        embedding_backend="onnx",  # different backend -> must rebuild
    )
    assert second.count() == 0, "stale collection was reused under a new backend"

    second.ingest_file(data_dir / "sample_policy.md")
    assert second.count() == 2


# -- semantics (opt-in) ----------------------------------------------------


def _onnx_model_is_cached() -> bool:
    try:
        from app.rag.embeddings import build_embedding_function

        build_embedding_function("onnx")(["warm"])
        return True
    except Exception:  # noqa: BLE001 - no model, no network, any failure
        return False


@pytest.mark.integration
@pytest.mark.skipif(
    not _onnx_model_is_cached(),
    reason="all-MiniLM-L6-v2 is not cached; run scripts/warm_embeddings.py",
)
def test_real_model_understands_paraphrase(tmp_path: Path, data_dir: Path) -> None:
    """The claim the lexical backend cannot make: retrieval by *meaning*.

    'A pipe exploded and soaked my floor' shares almost no vocabulary with the
    policy text, so only a semantic model gets this right.
    """
    r = PolicyRetriever(
        persist_dir=tmp_path / "chroma",
        collection_name="semantic",
        embedding_backend="onnx",
        top_k=2,
    )
    r.ingest_file(data_dir / "sample_policy.md")

    hits = r.search("A pipe exploded and soaked my floor. Am I protected?")
    assert hits[0].section_title == "Section 1: Home Water Damage Coverage"

    hits = r.search("My necklace was stolen, will you pay for it?")
    assert hits[0].section_title == "Section 2: Personal Property Protection"

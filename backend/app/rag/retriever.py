"""Policy retrieval over a local Chroma collection.

Design notes worth defending in review:

* **No distance threshold.** It is tempting to refuse when the best distance
  exceeds some cutoff, but the right cutoff depends on the embedding model and
  the corpus, and this corpus is two paragraphs. A hand-picked number would
  either reject legitimate questions during the demo or let anything through.
  Grounding is enforced where it belongs — in the prompt, which forbids
  answering outside the retrieved passages — and the resulting behaviour is
  asserted by tests rather than by a magic constant. Distances are still
  returned so the UI can show them.

* **Upsert on stable IDs.** Restarting the API re-ingests the document; stable
  chunk IDs mean that is idempotent instead of accumulating duplicates.

* **Backend recorded in collection metadata.** Switching embedding backends
  changes the vector dimension, which would otherwise fail at query time with a
  confusing error. We detect the mismatch and rebuild.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.rag.embeddings import build_embedding_function
from app.rag.splitter import (
    DEFAULT_MAX_CHUNK_CHARS,
    PolicyChunk,
    load_policy_chunks,
)

logger = logging.getLogger(__name__)

_BACKEND_KEY = "embedding_backend"


@dataclass(frozen=True)
class RetrievedPassage:
    """One passage returned by a search, ready to be cited."""

    text: str
    citation: str
    section_title: str
    source_file: str
    distance: float

    @property
    def relevance(self) -> float:
        """Cosine similarity in ``[-1, 1]``; higher is closer."""
        return round(1.0 - self.distance, 4)


class PolicyRetriever:
    """Ingest and search policy documents in a local Chroma collection."""

    def __init__(
        self,
        *,
        persist_dir: Path | str,
        collection_name: str = "omnicare_policy",
        embedding_backend: str = "onnx",
        top_k: int = 2,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    ) -> None:
        self._persist_dir = Path(persist_dir)
        self._collection_name = collection_name
        self._backend = embedding_backend
        self._top_k = top_k
        self._max_chunk_chars = max_chunk_chars

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._embedding_fn = build_embedding_function(embedding_backend)
        self._collection = self._open_collection()

    # -- ingestion ---------------------------------------------------------

    def ingest(self, chunks: list[PolicyChunk]) -> int:
        """Upsert chunks. Returns the number written."""
        if not chunks:
            logger.warning("No chunks to ingest; the policy document is empty.")
            return 0
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.embedding_text for c in chunks],
            metadatas=[{**c.as_chroma_metadata(), "raw_text": c.text} for c in chunks],
        )
        logger.info("Ingested %d policy chunks into %s", len(chunks), self._collection_name)
        return len(chunks)

    def ingest_file(self, path: Path | str) -> int:
        return self.ingest(
            load_policy_chunks(path, max_chunk_chars=self._max_chunk_chars)
        )

    # -- search ------------------------------------------------------------

    def search(self, query: str, *, top_k: int | None = None) -> list[RetrievedPassage]:
        """Return the most relevant passages for ``query``."""
        query = (query or "").strip()
        if not query or self.count() == 0:
            return []

        k = min(top_k or self._top_k, self.count())
        result = self._collection.query(
            query_texts=[query],
            n_results=k,
            include=["metadatas", "distances"],
        )

        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        passages: list[RetrievedPassage] = []
        for meta, distance in zip(metadatas, distances, strict=False):
            meta = meta or {}
            passages.append(
                RetrievedPassage(
                    text=str(meta.get("raw_text", "")),
                    citation=str(meta.get("citation", "")),
                    section_title=str(meta.get("section_title", "")),
                    source_file=str(meta.get("source_file", "")),
                    distance=float(distance),
                )
            )
        return passages

    # -- housekeeping ------------------------------------------------------

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection."""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:  # noqa: BLE001 - absent collection is fine
            logger.debug("Collection %s did not exist", self._collection_name)
        self._collection = self._create_collection()

    # -- internals ---------------------------------------------------------

    def _open_collection(self):
        try:
            existing = self._client.get_collection(
                name=self._collection_name, embedding_function=self._embedding_fn
            )
        except Exception:  # noqa: BLE001 - Chroma raises several types here
            return self._create_collection()

        recorded = (existing.metadata or {}).get(_BACKEND_KEY)
        if recorded != self._backend:
            logger.warning(
                "Collection %s was built with embedding backend %r but %r is "
                "configured; rebuilding to avoid a dimension mismatch.",
                self._collection_name,
                recorded,
                self._backend,
            )
            self._client.delete_collection(self._collection_name)
            return self._create_collection()
        return existing

    def _create_collection(self):
        return self._client.create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine", _BACKEND_KEY: self._backend},
        )

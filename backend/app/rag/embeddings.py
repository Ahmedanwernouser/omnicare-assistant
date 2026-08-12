"""Embedding backends for the policy vector store.

Two implementations behind one interface, chosen by ``Settings.embedding_backend``:

``onnx`` (default, production)
    Chroma's bundled all-MiniLM-L6-v2. No API key and no paid service, which is
    what the zero-cost requirement asks for. **The model weights are downloaded
    on first use (~80 MB).** The Dockerfile warms this cache at *build* time —
    see ``scripts/warm_embeddings.py``. Leaving it to runtime means the first
    question a reviewer asks either hangs for a minute or fails outright on a
    restricted network.

``lexical`` (tests, CI)
    A deterministic hashed bag-of-words embedding. Zero downloads, zero
    randomness, milliseconds to run. It is a genuinely weaker retriever — it
    matches shared vocabulary, not meaning — so it is used to prove the
    *plumbing* (ingest, query, metadata, citations) while the semantic quality
    of the real model is asserted separately in the integration test.

Keeping the backend injectable is what lets ``pytest`` run offline and keyless.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

LEXICAL_DIM = 512
_TOKEN = re.compile(r"[a-z0-9$]+")


class EmbeddingBackend(Protocol):
    def __call__(self, input: Documents) -> Embeddings: ...


class LexicalEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic hashed bag-of-words vectors, L2-normalised.

    Same text in, same vector out, on every machine and every run — so a
    retrieval assertion can never flake.
    """

    def __init__(self, dim: int = LEXICAL_DIM) -> None:
        self._dim = dim

    @staticmethod
    def name() -> str:
        return "lexical-hash"

    def get_config(self) -> dict[str, int]:
        return {"dim": self._dim}

    @staticmethod
    def build_from_config(config: dict[str, int]) -> LexicalEmbeddingFunction:
        return LexicalEmbeddingFunction(dim=config.get("dim", LEXICAL_DIM))

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            # Chroma rejects all-zero vectors; park empty documents on one axis.
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]


def build_embedding_function(backend: str) -> EmbeddingFunction[Documents]:
    """Return the embedding function named by ``backend``."""
    normalised = backend.strip().lower()
    if normalised == "lexical":
        return LexicalEmbeddingFunction()
    if normalised == "onnx":
        # Imported lazily: this triggers the model download on first use.
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        return ONNXMiniLM_L6_V2()
    raise ValueError(
        f"Unknown embedding backend {backend!r}. Expected 'onnx' or 'lexical'."
    )

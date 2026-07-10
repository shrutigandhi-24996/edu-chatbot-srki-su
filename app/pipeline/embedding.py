"""Shared SentenceTransformer loader.

Loading the embedding model once and reusing it across the intent classifier and
the retriever roughly halves memory use compared with loading it twice.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def get_shared_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model only once.
    """
    return SentenceTransformer(
        EMBEDDING_MODEL
    )


def embed_texts(
    texts: list[str],
) -> np.ndarray:
    """
    Generate normalized embeddings for multiple texts.
    """
    if not texts:
        return np.empty(
            shape=(0, 0),
            dtype=np.float32,
        )

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 20,
    )

    return embeddings.astype(
        np.float32
    )


def embed_query(
    query: str,
) -> np.ndarray:
    """
    Generate one normalized query embedding.
    """
    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    return embed_texts([query])[0]
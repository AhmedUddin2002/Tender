from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: Iterable[str], normalize: bool = True) -> np.ndarray:
    model = _model()
    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return embeddings


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError
import pandas as pd

from config import VECTOR_DB_PATH
from embeddings import embed_query, embed_texts


_COLLECTION_NAME = "tenders"


def _client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(VECTOR_DB_PATH))


def _get_collection(reset: bool = False) -> Collection:
    client = _client()
    if reset:
        try:
            client.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass
    try:
        return client.get_collection(_COLLECTION_NAME)
    except (ValueError, NotFoundError):
        return client.create_collection(name=_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def rebuild_collection(df: pd.DataFrame) -> None:
    collection = _get_collection(reset=True)
    if df.empty:
        return

    documents = []
    metadatas = []
    ids = []
    for idx, row in df.iterrows():
        reference = row.get("reference_no") or f"tender_{idx}"
        text = "\n".join(
            filter(
                None,
                [
                    str(row.get("tender_title", "")),
                    str(row.get("description", "")),
                    f"Buyer: {row.get('buyer_name', '')}",
                    f"Category: {row.get('category', '')}",
                ],
            )
        )
        documents.append(text)
        metadatas.append(
            {
                "reference_no": reference,
                "tender_title": row.get("tender_title"),
                "source": row.get("source"),
                "category": row.get("category"),
            }
        )
        ids.append(str(reference))

    embeddings = embed_texts(documents)
    collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings.tolist())


@dataclass
class VectorResult:
    score: float
    reference_no: str
    tender_title: str
    source: Optional[str]
    category: Optional[str]
    text: str


def semantic_search(query: str, top_k: int = 10) -> List[VectorResult]:
    collection = _get_collection()
    query_vector = embed_query(query)
    result = collection.query(query_embeddings=[query_vector.tolist()], n_results=top_k)
    distances = result.get("distances", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    documents = result.get("documents", [[]])[0]

    vector_results: List[VectorResult] = []
    for distance, metadata, document in zip(distances, metadatas, documents):
        similarity = 1 - distance if distance is not None else 0.0
        vector_results.append(
            VectorResult(
                score=float(max(0.0, min(1.0, similarity))),
                reference_no=metadata.get("reference_no", ""),
                tender_title=metadata.get("tender_title", ""),
                source=metadata.get("source"),
                category=metadata.get("category"),
                text=document,
            )
        )
    return vector_results

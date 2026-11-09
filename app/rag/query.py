import os
from typing import Any

from qdrant_client import models

from app.rag.hybrid_embedding import get_hybrid_encoder
from app.rag.qdrant_index import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    get_qdrant_client,
)


DEFAULT_PREFETCH_LIMIT = 50
DEFAULT_RRF_K = 60


def _positive_env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    value = int(raw_value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _law_filter(law_name: str | None) -> models.Filter | None:
    if not law_name:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="law_name",
                match=models.MatchValue(value=law_name),
            )
        ]
    )


def retrieve_laws(
    query: str,
    top_k: int = 5,
    law_name: str | None = None,
) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    embedding = get_hybrid_encoder().encode_query(query)
    query_filter = _law_filter(law_name)
    prefetch_limit = max(
        top_k,
        _positive_env_int("HYBRID_PREFETCH_LIMIT", DEFAULT_PREFETCH_LIMIT),
    )
    rrf_k = _positive_env_int("HYBRID_RRF_K", DEFAULT_RRF_K)

    response = get_qdrant_client().query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=embedding.dense,
                using=DENSE_VECTOR_NAME,
                filter=query_filter,
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=embedding.sparse.indices,
                    values=embedding.sparse.values,
                ),
                using=SPARSE_VECTOR_NAME,
                filter=query_filter,
                limit=prefetch_limit,
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf(k=rrf_k)),
        limit=top_k,
        with_payload=["text", "law_name", "section", "source_file"],
        with_vectors=False,
    )

    results: list[dict[str, Any]] = []
    for point in response.points:
        payload = point.payload or {}
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        results.append(
            {
                "score": float(point.score),
                "text": text,
                "metadata": {
                    "law_name": payload.get("law_name"),
                    "section": payload.get("section"),
                    "source_file": payload.get("source_file"),
                },
            }
        )
    return results


if __name__ == "__main__":
    for result in retrieve_laws(
        query="นายจ้างต้องจ่ายค่าล่วงเวลาที่ไหน",
        top_k=3,
    ):
        print("Score:", result["score"])
        print("Law:", result["metadata"].get("law_name"))
        print("Section:", result["metadata"].get("section"))
        print("Text:", result["text"])
        print("-" * 50)

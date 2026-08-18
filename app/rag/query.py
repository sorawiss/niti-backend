from typing import Any

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore

from app.rag.embedding import get_embed_model
from app.rag.qdrant_index import COLLECTION_NAME, get_qdrant_client


def get_index() -> VectorStoreIndex:
    vector_store = QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=COLLECTION_NAME,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=get_embed_model(),
    )


def retrieve_laws(
    query: str,
    top_k: int = 5,
    law_name: str | None = None,
) -> list[dict[str, Any]]:
    index = get_index()

    filters_list: list[MetadataFilter] = []

    if law_name:
        filters_list.append(
            MetadataFilter(
                key="law_name",
                value=law_name,
                operator=FilterOperator.EQ,
            )
        )

    filters = MetadataFilters(filters=filters_list) if filters_list else None

    retriever = index.as_retriever(
        similarity_top_k=top_k,
        filters=filters,
    )

    results: list[NodeWithScore] = retriever.retrieve(query)

    return [
        {
            "score": result.score,
            "text": result.node.get_content(),
            "metadata": result.node.metadata,
        }
        for result in results
    ]


if __name__ == "__main__":
    results = retrieve_laws(
        query="นายจ้างต้องจ่ายค่าล่วงเวลาที่ไหน",
        top_k=3,
    )

    for result in results:
        print("Score:", result["score"])
        print("Law:", result["metadata"].get("law_name"))
        print("Section:", result["metadata"].get("section"))
        print("Topic:", result["metadata"].get("topic"))
        print("Text:", result["text"])
        print("-" * 50)

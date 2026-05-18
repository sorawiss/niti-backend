import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.rag.embedding import get_embed_model

load_dotenv()

COLLECTION_NAME = "thai_labor_law"
DEFAULT_CHUNKS_DIR = Path("data/law-chunks")
REQUIRED_FIELDS = ("law_name", "section", "topic", "text")


def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        raise ValueError("QDRANT_URL is not set")

    return QdrantClient(
        url=url,
        api_key=api_key,
        cloud_inference=True,
    )


def load_chunks(chunks_dir: Path = DEFAULT_CHUNKS_DIR) -> list[dict[str, Any]]:
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunk directory does not exist: {chunks_dir}")

    chunks: list[dict[str, Any]] = []

    for file_path in sorted(chunks_dir.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as file:
            chunk = json.load(file)

        missing_fields = [field for field in REQUIRED_FIELDS if not chunk.get(field)]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(f"{file_path} is missing required fields: {missing}")

        chunk["source_file"] = file_path.name
        chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No JSON chunks found in {chunks_dir}")

    return chunks


def build_index(chunks: list[dict[str, Any]]) -> VectorStoreIndex:
    vector_store = QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=COLLECTION_NAME,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = [
        Document(
            text=chunk["text"],
            metadata={
                "law_name": chunk["law_name"],
                "section": chunk["section"],
                "topic": chunk["topic"],
                "source_file": chunk.get("source_file"),
            },
        )
        for chunk in chunks
    ]

    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=get_embed_model(),
    )


def main() -> None:
    chunks = load_chunks()
    build_index(chunks)
    print(f"Indexed {len(chunks)} chunks into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from app.rag.hybrid_embedding import MODEL_NAME, BgeM3HybridEncoder, get_hybrid_encoder

load_dotenv()

LEGACY_COLLECTION_NAME = "thai_labor_law"
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "thai_labor_law_hybrid")
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_VECTOR_SIZE = 1024
DEFAULT_CHUNKS_DIR = Path("data/law-chunks")
DEFAULT_INDEX_BATCH_SIZE = 8
REQUIRED_FIELDS = ("law_name", "section", "text")
PAYLOAD_INDEX_FIELDS = ("law_name", "section", "source_file", "content_hash")


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url:
        raise ValueError("QDRANT_URL is not set")
    return QdrantClient(url=url, api_key=api_key)


def load_chunks(chunks_dir: Path = DEFAULT_CHUNKS_DIR) -> list[dict[str, Any]]:
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunk directory does not exist: {chunks_dir}")

    chunks: list[dict[str, Any]] = []
    for file_path in sorted(chunks_dir.rglob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at {file_path}:{line_number} - {error}"
                    ) from error

                missing_fields = [field for field in REQUIRED_FIELDS if not chunk.get(field)]
                if missing_fields:
                    missing = ", ".join(missing_fields)
                    raise ValueError(
                        f"{file_path}:{line_number} is missing required fields: {missing}"
                    )

                chunk["source_file"] = file_path.relative_to(chunks_dir).as_posix()
                chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No JSONL chunks found in {chunks_dir}")
    return chunks


def _validate_collection(client: QdrantClient) -> None:
    info = client.get_collection(COLLECTION_NAME)
    dense_vectors = info.config.params.vectors
    sparse_vectors = info.config.params.sparse_vectors or {}
    if not isinstance(dense_vectors, dict) or DENSE_VECTOR_NAME not in dense_vectors:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' does not contain named dense vector "
            f"'{DENSE_VECTOR_NAME}'"
        )
    if dense_vectors[DENSE_VECTOR_NAME].size != DENSE_VECTOR_SIZE:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' has an incompatible dense vector size"
        )
    if SPARSE_VECTOR_NAME not in sparse_vectors:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' does not contain sparse vector "
            f"'{SPARSE_VECTOR_NAME}'"
        )


def ensure_collection(client: QdrantClient, *, recreate: bool = False) -> None:
    exists = client.collection_exists(COLLECTION_NAME)
    if recreate and exists:
        if COLLECTION_NAME == LEGACY_COLLECTION_NAME:
            raise ValueError("Refusing to recreate the legacy collection")
        client.delete_collection(COLLECTION_NAME)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(),
            },
            on_disk_payload=True,
        )
    else:
        _validate_collection(client)

    payload_schema = client.get_collection(COLLECTION_NAME).payload_schema
    for field_name in PAYLOAD_INDEX_FIELDS:
        if field_name not in payload_schema:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )


def _point_id(chunk: dict[str, Any]) -> str:
    identity = json.dumps(
        [chunk["law_name"], chunk["section"], chunk["text"]],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return str(uuid5(NAMESPACE_URL, identity))


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_points(
    chunks: list[dict[str, Any]],
    encoder: BgeM3HybridEncoder,
) -> list[models.PointStruct]:
    embeddings = encoder.encode_documents([str(chunk["text"]) for chunk in chunks])
    points: list[models.PointStruct] = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        text = str(chunk["text"])
        points.append(
            models.PointStruct(
                id=_point_id(chunk),
                vector={
                    DENSE_VECTOR_NAME: embedding.dense,
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=embedding.sparse.indices,
                        values=embedding.sparse.values,
                    ),
                },
                payload={
                    "text": text,
                    "law_name": chunk["law_name"],
                    "section": chunk["section"],
                    "source_file": chunk.get("source_file"),
                    "content_hash": _content_hash(text),
                    "embedding_model": MODEL_NAME,
                    "index_version": 2,
                },
            )
        )
    return points


def build_index(
    chunks: list[dict[str, Any]],
    *,
    client: QdrantClient | None = None,
    encoder: BgeM3HybridEncoder | None = None,
    recreate: bool = False,
    batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if COLLECTION_NAME == LEGACY_COLLECTION_NAME:
        raise ValueError("Hybrid indexing must not target the legacy collection")

    qdrant = client or get_qdrant_client()
    hybrid_encoder = encoder or get_hybrid_encoder()
    ensure_collection(qdrant, recreate=recreate)

    indexed = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=_build_points(batch, hybrid_encoder),
            wait=True,
        )
        indexed += len(batch)
        print(f"Indexed {indexed}/{len(chunks)} chunks", flush=True)
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the BGE-M3 dense+sparse Qdrant collection."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and rebuild only the hybrid collection before indexing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_INDEX_BATCH_SIZE,
    )
    args = parser.parse_args()

    chunks = load_chunks()
    indexed = build_index(
        chunks,
        recreate=args.recreate,
        batch_size=args.batch_size,
    )
    print(f"Indexed {indexed} chunks into '{COLLECTION_NAME}'")
    print(f"Legacy collection '{LEGACY_COLLECTION_NAME}' was not modified")


if __name__ == "__main__":
    main()

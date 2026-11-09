import unittest

from qdrant_client import QdrantClient

from app.rag.hybrid_embedding import HybridEmbedding, SparseEmbedding
from app.rag.qdrant_index import (
    COLLECTION_NAME,
    DENSE_VECTOR_SIZE,
    LEGACY_COLLECTION_NAME,
    build_index,
)


class _FakeEncoder:
    def encode_documents(self, texts: list[str]) -> list[HybridEmbedding]:
        return [
            HybridEmbedding(
                dense=[0.0] * DENSE_VECTOR_SIZE,
                sparse=SparseEmbedding(indices=[index + 1], values=[1.0]),
            )
            for index, _text in enumerate(texts)
        ]


class QdrantHybridIndexTests(unittest.TestCase):
    def test_builds_named_vectors_and_is_idempotent(self) -> None:
        client = QdrantClient(location=":memory:")
        chunks = [
            {
                "law_name": "กฎหมายแรงงาน",
                "section": "มาตรา 1",
                "text": "ข้อความหนึ่ง",
                "source_file": "chapter.jsonl",
            },
            {
                "law_name": "กฎหมายแรงงาน",
                "section": "มาตรา 2",
                "text": "ข้อความสอง",
                "source_file": "chapter.jsonl",
            },
        ]

        build_index(chunks, client=client, encoder=_FakeEncoder(), batch_size=2)
        build_index(chunks, client=client, encoder=_FakeEncoder(), batch_size=1)

        info = client.get_collection(COLLECTION_NAME)
        points, _ = client.scroll(
            COLLECTION_NAME,
            limit=10,
            with_payload=True,
            with_vectors=True,
        )
        self.assertEqual(client.count(COLLECTION_NAME, exact=True).count, 2)
        self.assertIn("dense", info.config.params.vectors)
        self.assertIn("sparse", info.config.params.sparse_vectors)
        self.assertEqual(points[0].payload["embedding_model"], "BAAI/bge-m3")
        self.assertIn("dense", points[0].vector)
        self.assertIn("sparse", points[0].vector)
        self.assertFalse(client.collection_exists(LEGACY_COLLECTION_NAME))


if __name__ == "__main__":
    unittest.main()

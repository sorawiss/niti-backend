import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qdrant_client import QdrantClient

from app.rag.hybrid_embedding import HybridEmbedding, SparseEmbedding
from app.rag.qdrant_index import (
    COLLECTION_NAME,
    DENSE_VECTOR_SIZE,
    LEGACY_COLLECTION_NAME,
    build_index,
    list_indexed_law_names,
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


    def test_law_catalog_reads_every_page_and_preserves_exact_values(self) -> None:
        with patch("app.rag.qdrant_index.get_qdrant_client") as get_client:
            client = get_client.return_value
            client.scroll.side_effect = [
                ([SimpleNamespace(payload={"law_name": " Act A "})], 123),
                ([
                    SimpleNamespace(payload={"law_name": " Act A "}),
                    SimpleNamespace(payload={"law_name": "Act B"}),
                    SimpleNamespace(payload={}),
                ], None),
            ]

            self.assertEqual(list_indexed_law_names(), (" Act A ", "Act B"))
            self.assertEqual(
                [call.kwargs["offset"] for call in client.scroll.call_args_list],
                [None, 123],
            )
            self.assertEqual(client.scroll.call_args.kwargs["with_payload"], ["law_name"])
            self.assertFalse(client.scroll.call_args.kwargs["with_vectors"])


if __name__ == "__main__":
    unittest.main()

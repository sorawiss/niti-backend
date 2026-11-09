import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qdrant_client import models

from app.rag.hybrid_embedding import HybridEmbedding, SparseEmbedding
from app.rag.query import retrieve_laws


class _FakeEncoder:
    def encode_query(self, text: str) -> HybridEmbedding:
        return HybridEmbedding(
            dense=[0.1, 0.2],
            sparse=SparseEmbedding(indices=[7, 11], values=[0.8, 0.4]),
        )


class _FakeClient:
    def __init__(self) -> None:
        self.kwargs = None

    def query_points(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=0.032,
                    payload={
                        "text": "ข้อความกฎหมาย",
                        "law_name": "กฎหมายแรงงาน",
                        "section": "มาตรา 1",
                        "source_file": "chapter.jsonl",
                    },
                )
            ]
        )


class HybridQueryTests(unittest.TestCase):
    @patch("app.rag.query.get_hybrid_encoder", return_value=_FakeEncoder())
    @patch("app.rag.query.get_qdrant_client")
    def test_uses_native_rrf_and_preserves_result_contract(
        self,
        get_client,
        _get_encoder,
    ) -> None:
        client = _FakeClient()
        get_client.return_value = client

        results = retrieve_laws("ค่าล่วงเวลา", top_k=3, law_name="กฎหมายแรงงาน")

        self.assertEqual(results[0]["text"], "ข้อความกฎหมาย")
        self.assertEqual(results[0]["metadata"]["section"], "มาตรา 1")
        self.assertEqual(client.kwargs["limit"], 3)
        self.assertIsInstance(client.kwargs["query"], models.RrfQuery)
        self.assertEqual(client.kwargs["query"].rrf.k, 60)
        self.assertEqual(len(client.kwargs["prefetch"]), 2)
        self.assertEqual(client.kwargs["prefetch"][0].using, "dense")
        self.assertEqual(client.kwargs["prefetch"][1].using, "sparse")
        self.assertIsNotNone(client.kwargs["prefetch"][0].filter)
        self.assertIsNotNone(client.kwargs["prefetch"][1].filter)

    def test_rejects_invalid_input_before_loading_models(self) -> None:
        with self.assertRaises(ValueError):
            retrieve_laws(" ")
        with self.assertRaises(ValueError):
            retrieve_laws("query", top_k=0)


if __name__ == "__main__":
    unittest.main()

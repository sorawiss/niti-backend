import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qdrant_client import models

from app.rag.hybrid_embedding import HybridEmbedding, SparseEmbedding
from app.rag.query import retrieve_laws
from app.rag.reranker import RerankerSettings


class _FakeEncoder:
    def encode_query(self, text: str) -> HybridEmbedding:
        return HybridEmbedding(
            dense=[0.1, 0.2],
            sparse=SparseEmbedding(indices=[7, 11], values=[0.8, 0.4]),
        )


class _FakeClient:
    def __init__(self, points=None) -> None:
        self.kwargs = None
        self.points = points

    def query_points(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            points=self.points or [
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


class _FakeReranker:
    def __init__(self, scores=None, error=None) -> None:
        self.scores = scores or []
        self.error = error
        self.calls = []

    def score(self, query, passages):
        self.calls.append((query, passages))
        if self.error:
            raise self.error
        return self.scores


def _point(text: str, score: float, section: str):
    return SimpleNamespace(
        score=score,
        payload={
            "text": text,
            "law_name": "law",
            "section": section,
            "source_file": "chapter.jsonl",
        },
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

    @patch("app.rag.query.get_hybrid_encoder", return_value=_FakeEncoder())
    @patch("app.rag.query.get_qdrant_client")
    @patch("app.rag.query.get_reranker")
    @patch("app.rag.query.get_reranker_settings")
    def test_reranks_expanded_candidates_and_preserves_provenance(
        self,
        get_settings,
        get_reranker,
        get_client,
        _get_encoder,
    ) -> None:
        get_settings.return_value = RerankerSettings(
            enabled=True,
            candidate_limit=3,
        )
        reranker = _FakeReranker([0.1, 0.9, 0.8])
        get_reranker.return_value = reranker
        client = _FakeClient(
            [
                _point("first", 0.03, "1"),
                _point("second", 0.02, "2"),
                _point("third", 0.01, "3"),
            ]
        )
        get_client.return_value = client

        results = retrieve_laws("query", top_k=2)

        self.assertEqual(client.kwargs["limit"], 3)
        self.assertEqual([result["text"] for result in results], ["second", "third"])
        self.assertEqual(results[0]["score"], 0.9)
        self.assertEqual(results[0]["rerank_score"], 0.9)
        self.assertEqual(results[0]["retrieval_score"], 0.02)
        self.assertEqual(results[0]["retrieval_rank"], 2)
        self.assertEqual(results[0]["rerank_rank"], 1)
        self.assertEqual(reranker.calls, [("query", ["first", "second", "third"])])

    @patch("app.rag.query.get_hybrid_encoder", return_value=_FakeEncoder())
    @patch("app.rag.query.get_qdrant_client")
    @patch("app.rag.query.get_reranker")
    @patch("app.rag.query.get_reranker_settings")
    def test_reranker_ties_keep_retrieval_order(
        self,
        get_settings,
        get_reranker,
        get_client,
        _get_encoder,
    ) -> None:
        get_settings.return_value = RerankerSettings(
            enabled=True,
            candidate_limit=2,
        )
        get_reranker.return_value = _FakeReranker([0.5, 0.5])
        get_client.return_value = _FakeClient(
            [_point("first", 0.03, "1"), _point("second", 0.02, "2")]
        )

        results = retrieve_laws("query", top_k=2)

        self.assertEqual([result["text"] for result in results], ["first", "second"])

    @patch("app.rag.query.get_hybrid_encoder", return_value=_FakeEncoder())
    @patch("app.rag.query.get_qdrant_client")
    @patch("app.rag.query.get_reranker")
    @patch("app.rag.query.get_reranker_settings")
    def test_reranker_failure_respects_fail_open(
        self,
        get_settings,
        get_reranker,
        get_client,
        _get_encoder,
    ) -> None:
        get_settings.return_value = RerankerSettings(
            enabled=True,
            fail_open=True,
            candidate_limit=2,
        )
        get_reranker.return_value = _FakeReranker(error=RuntimeError("failed"))
        get_client.return_value = _FakeClient(
            [_point("first", 0.03, "1"), _point("second", 0.02, "2")]
        )

        with self.assertLogs("uvicorn.error", level="INFO"):
            results = retrieve_laws("query", top_k=1)

        self.assertEqual([result["text"] for result in results], ["first"])
        self.assertIsNone(results[0]["rerank_score"])

        get_settings.return_value = RerankerSettings(
            enabled=True,
            fail_open=False,
            candidate_limit=2,
        )
        with self.assertLogs("uvicorn.error", level="INFO"):
            with self.assertRaises(RuntimeError):
                retrieve_laws("query", top_k=1)

    def test_rejects_invalid_input_before_loading_models(self) -> None:
        with self.assertRaises(ValueError):
            retrieve_laws(" ")
        with self.assertRaises(ValueError):
            retrieve_laws("query", top_k=0)


if __name__ == "__main__":
    unittest.main()

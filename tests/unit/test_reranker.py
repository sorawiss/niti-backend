import os
import unittest
from unittest.mock import patch

from app.rag.reranker import (
    BgeReranker,
    get_reranker_settings,
)


class _FakeModel:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def compute_score(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return self.scores


class BgeRerankerTests(unittest.TestCase):
    def test_scores_pairs_with_normalization_and_batching(self) -> None:
        model = _FakeModel([0.2, 0.9])
        reranker = BgeReranker(model, batch_size=4)

        scores = reranker.score("query", ["first", "second"])

        self.assertEqual(scores, [0.2, 0.9])
        pairs, kwargs = model.calls[0]
        self.assertEqual(pairs, [["query", "first"], ["query", "second"]])
        self.assertEqual(kwargs, {"batch_size": 4, "normalize": True})

    def test_accepts_scalar_for_one_passage(self) -> None:
        reranker = BgeReranker(_FakeModel(0.75))

        self.assertEqual(reranker.score("query", ["passage"]), [0.75])

    def test_rejects_invalid_model_output(self) -> None:
        cases = [([0.2], ["one", "two"]), ([1.1], ["one"])]
        for scores, passages in cases:
            with self.subTest(scores=scores):
                with self.assertRaises(RuntimeError):
                    BgeReranker(_FakeModel(scores)).score("query", passages)

    def test_rejects_empty_inputs_before_model_call(self) -> None:
        model = _FakeModel([])
        reranker = BgeReranker(model)

        with self.assertRaises(ValueError):
            reranker.score(" ", ["passage"])
        with self.assertRaises(ValueError):
            reranker.score("query", [" "])
        self.assertEqual(reranker.score("query", []), [])
        self.assertEqual(model.calls, [])


class RerankerSettingsTests(unittest.TestCase):
    def tearDown(self) -> None:
        get_reranker_settings.cache_clear()

    def test_defaults_are_disabled_and_fail_open(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            get_reranker_settings.cache_clear()
            settings = get_reranker_settings()

        self.assertFalse(settings.enabled)
        self.assertTrue(settings.fail_open)
        self.assertEqual(settings.candidate_limit, 25)

    def test_environment_configuration_is_validated(self) -> None:
        environment = {
            "RERANKER_ENABLED": "true",
            "RERANKER_FAIL_OPEN": "false",
            "RERANKER_CANDIDATE_LIMIT": "40",
            "RERANKER_BATCH_SIZE": "3",
        }
        with patch.dict(os.environ, environment, clear=True):
            get_reranker_settings.cache_clear()
            settings = get_reranker_settings()

        self.assertTrue(settings.enabled)
        self.assertFalse(settings.fail_open)
        self.assertEqual(settings.candidate_limit, 40)
        self.assertEqual(settings.batch_size, 3)


if __name__ == "__main__":
    unittest.main()

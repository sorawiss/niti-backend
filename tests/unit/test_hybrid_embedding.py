import unittest

from app.rag.hybrid_embedding import BgeM3HybridEncoder


class _DenseVector(list):
    def tolist(self):
        return list(self)


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": texts, **kwargs})
        return {
            "dense_vecs": [_DenseVector([0.1, 0.2]) for _ in texts],
            "lexical_weights": [
                {"9": 0.4, "2": 0.8, "5": 0.0} for _ in texts
            ],
        }


class BgeM3HybridEncoderTests(unittest.TestCase):
    def test_generates_dense_and_sorted_sparse_vectors_in_one_call(self) -> None:
        model = _FakeModel()
        encoder = BgeM3HybridEncoder(model, batch_size=2, max_length=1024)

        results = encoder.encode_documents(["หนึ่ง", "สอง"])

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["return_dense"], True)
        self.assertEqual(model.calls[0]["return_sparse"], True)
        self.assertEqual(model.calls[0]["return_colbert_vecs"], False)
        self.assertEqual(results[0].dense, [0.1, 0.2])
        self.assertEqual(results[0].sparse.indices, [2, 9])
        self.assertEqual(results[0].sparse.values, [0.8, 0.4])

    def test_rejects_empty_query(self) -> None:
        encoder = BgeM3HybridEncoder(_FakeModel())
        with self.assertRaises(ValueError):
            encoder.encode_query("   ")


if __name__ == "__main__":
    unittest.main()

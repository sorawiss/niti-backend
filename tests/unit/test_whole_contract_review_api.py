import unittest

from fastapi.testclient import TestClient

from app.api.reviews import (
    get_clause_retrieval_service,
    get_whole_contract_review_service,
)
from app.main import app
from app.llm.config import QueryRewriteWorkflow
from app.reviews.whole_contract_reviewer import WholeContractReviewService
from tests.unit.test_clause_reviewer import (
    FakeGateway,
    make_clause_result,
    make_preparation,
)
from tests.unit.test_contract_review_api import make_docx
from tests.unit.test_whole_contract_reviewer import whole_review_json


client = TestClient(app)


class FakeWholeContractRetrievalService:
    def __init__(self) -> None:
        self.top_k: int | None = None
        self.workflow = None

    async def prepare(self, document, *, top_k: int = 5, workflow=None):
        self.top_k = top_k
        self.workflow = workflow
        return make_preparation([make_clause_result("clause-0001")])


class WholeContractReviewApiTests(unittest.TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_exposes_validated_whole_contract_review_for_frontend(self) -> None:
        retrieval = FakeWholeContractRetrievalService()
        gateway = FakeGateway([whole_review_json(["clause-0001"])])
        review_service = WholeContractReviewService(gateway)
        app.dependency_overrides[get_clause_retrieval_service] = lambda: retrieval
        app.dependency_overrides[get_whole_contract_review_service] = (
            lambda: review_service
        )

        response = client.post(
            "/reviews/whole-contract",
            files={
                "file": (
                    "contract.docx",
                    make_docx(),
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document",
                )
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            retrieval.workflow,
            QueryRewriteWorkflow.WHOLE_CONTRACT,
        )
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(retrieval.top_k, 5)
        self.assertEqual(len(gateway.requests), 1)
        self.assertEqual(len(payload["completeness"]), 8)
        self.assertEqual(
            set(payload["completeness"][0]),
            {"category", "label", "status", "summary"},
        )
        self.assertEqual(
            payload["clause_reviews"][0]["assessments"][0][
                "supporting_contract_clause_ids"
            ],
            ["clause-0001"],
        )
        assessment_payload = payload["clause_reviews"][0]["assessments"][0]
        self.assertNotIn("evidence_gap", assessment_payload)
        self.assertNotIn("suggested_revision", assessment_payload)
        self.assertEqual(
            payload["unique_legal_sources"][0]["source_id"],
            "law:overtime",
        )
        self.assertEqual(
            payload["evidence_mappings"][0]["clause_id"],
            "clause-0001",
        )
        self.assertEqual(
            payload["clause_reviews"][0]["query_results"][0]["rewrite"]["status"],
            "disabled",
        )
        self.assertIn("document_blocks", payload)
        self.assertNotIn("llm_debug", payload)
        self.assertNotIn("raw_response", payload)


if __name__ == "__main__":
    unittest.main()

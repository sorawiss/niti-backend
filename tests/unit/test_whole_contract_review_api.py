import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.reviews import (
    get_clause_retrieval_service,
    get_whole_contract_review_service,
)
from app.main import app
from app.llm.config import QueryRewriteWorkflow
from app.reviews.retrieval import ClauseRetriever
from app.reviews.whole_contract_workflow import WholeContractReviewWorkflow
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

    async def prepare(self, document, *, top_k: int = 5, workflow=None, rewrite_queries=True):
        self.top_k = top_k
        self.workflow = workflow
        self.rewrite_queries = rewrite_queries
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
        self.assertIsNone(payload["contract_findings"])
        self.assertEqual(payload["overall_summary"], "สรุปผลการตรวจสัญญาโดยรวม")
        self.assertEqual(retrieval.top_k, 5)
        self.assertFalse(retrieval.rewrite_queries)
        self.assertEqual(len(gateway.requests), 1)
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
        self.assertNotIn("workflow_debug", payload)
        self.assertNotIn("llm_debug", payload)
        self.assertNotIn("raw_response", payload)


    def test_workflow_debug_traces_search_fallback_and_reviewer_input(self) -> None:
        clause_id = "clause-0001"
        search = {
            "clause_id": clause_id,
            "missing_issue": "Overtime exception",
            "query": "Section 24 emergency overtime exception",
            "law_name": "Labor Protection Act",
            "top_k": 2,
        }
        unknown = {**search, "law_name": "Unindexed law", "query": "Unindexed law exception"}
        gateway = FakeGateway([
            json.dumps({"searches": [search, search, unknown]}),
            whole_review_json([clause_id]),
        ])
        calls = []

        def retrieve(*, query, law_name, top_k):
            calls.append((query, law_name, top_k))
            if law_name:
                return []
            return [{
                "text": "Supplemental evidence about emergency work.",
                "metadata": {"law_name": "Labor Protection Act", "section": "24"},
            }]

        with patch.dict("os.environ", {"WHOLE_CONTRACT_PLANNER_MAX_OUTPUT_TOKENS": "9000"}):
            workflow = WholeContractReviewWorkflow(
                gateway,
                retriever=ClauseRetriever(retrieve),
                law_catalog=lambda: ("Labor Protection Act", "Another Indexed Act"),
            )
        app.dependency_overrides[get_clause_retrieval_service] = (
            FakeWholeContractRetrievalService
        )
        app.dependency_overrides[get_whole_contract_review_service] = lambda: workflow
        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            response = client.post(
                "/reviews/whole-contract?debug=true",
                files={"file": ("contract.docx", make_docx())},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(gateway.requests), 2)
        self.assertEqual(gateway.requests[0].max_output_tokens, 9000)
        self.assertEqual(gateway.requests[1].max_output_tokens, 16384)
        self.assertEqual(calls, [
            (search["query"], search["law_name"], 2),
            (search["query"], None, 2),
            (unknown["query"], None, 2),
        ])
        planner_prompt = gateway.requests[0].messages[1].content
        self.assertIn("law:overtime", planner_prompt)
        self.assertIn("Another Indexed Act", planner_prompt)
        self.assertNotIn("Another Indexed Act", gateway.requests[1].messages[1].content)
        self.assertIn("EVIDENCE_MAPPINGS", planner_prompt)
        self.assertNotIn("EVIDENCE_MAPPINGS", gateway.requests[1].messages[1].content)
        self.assertIn(
            "Supplemental evidence about emergency work.",
            gateway.requests[1].messages[1].content,
        )
        payload = response.json()
        trace = payload["workflow_debug"]
        prefix = "whole_contract_workflow_debug="
        logged_traces = [
            json.loads(record.getMessage()[len(prefix):])
            for record in logs.records
            if record.getMessage().startswith(prefix)
        ]
        self.assertEqual(logged_traces, [{
            "document_id": payload["document_id"],
            "workflow_debug": trace,
        }])
        self.assertEqual(trace["initial_retrieval"]["source_ids"], ["law:overtime"])
        self.assertEqual(trace["context_planning"]["searches"], [search, search, unknown])
        self.assertEqual(trace["context_planning"]["max_output_tokens"], 9000)
        lookup, duplicate, unfiltered = trace["supplemental_retrieval"]
        self.assertTrue(lookup["used_unfiltered_fallback"])
        self.assertEqual(lookup["status"], "ready")
        self.assertEqual(duplicate["reason"], "duplicate")
        self.assertEqual(lookup["applied_law_name"], "Labor Protection Act")
        self.assertEqual(unfiltered["unknown_law_name"], "Unindexed law")
        self.assertIsNone(unfiltered["applied_law_name"])
        self.assertEqual(trace["review_input"]["source_ids"], [
            "law:overtime", lookup["source_ids"][0],
        ])
        self.assertNotIn("evidence_mappings", trace["review_input"])
        self.assertEqual(
            trace["retrieval_provenance"]["evidence_mappings"],
            payload["evidence_mappings"],
        )
        self.assertTrue(trace["legal_review"]["accepted"])
        for step in ("context_planning", "legal_review"):
            self.assertEqual(trace[step]["model"], "test/model")
            self.assertEqual(trace[step]["usage"]["total_tokens"], 130)
            self.assertEqual(trace[step]["usage"]["cost"], 0.001)
        self.assertEqual(trace["total_llm_cost"], 0.002)
        self.assertNotIn("llm_debug", payload)


if __name__ == "__main__":
    unittest.main()

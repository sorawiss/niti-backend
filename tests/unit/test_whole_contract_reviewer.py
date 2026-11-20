import json
import unittest

from app.llm.models import LLMResponse, LLMUsage
from app.reviews.review_models import ClauseReviewStatus, ReviewRunStatus
from app.reviews.whole_contract_models import COMPLETENESS_CATEGORY_LABELS
from app.reviews.whole_contract_reviewer import WholeContractReviewService
from tests.unit.test_clause_reviewer import (
    FakeGateway,
    assessment,
    make_clause_result,
    make_preparation,
)

def whole_assessment(
    assessment_id: str,
    outcome: str,
    *,
    clause_id: str,
    source_id: str = "law:overtime",
) -> dict[str, object]:
    value = assessment(assessment_id, outcome, source_id=source_id)
    value.pop("evidence_gap")
    value.pop("suggested_revision")
    value["supporting_contract_clause_ids"] = [clause_id]
    return value



def completeness_items() -> list[dict[str, str]]:
    return [
        {
            "category": category.value,
            "label": label,
            "status": "complete",
            "summary": "\u0e1e\u0e1a\u0e02\u0e49\u0e2d\u0e01\u0e33\u0e2b\u0e19\u0e14\u0e04\u0e23\u0e1a\u0e16\u0e49\u0e27\u0e19",
        }
        for category, label in COMPLETENESS_CATEGORY_LABELS.items()
    ]


def whole_review_json(
    clause_ids: list[str],
    *,
    source_id: str = "law:overtime",
) -> str:
    return json.dumps(
        {
            "clause_reviews": [
                {
                    "clause_id": clause_id,
                    "applicable": True,
                    "applicability_reason": "มีข้อกำหนดการจ้างงาน",
                    "assessments": [
                        whole_assessment(
                            f"assessment-{index:02d}", "compliant",
                            clause_id=clause_id, source_id=source_id,
                        )
                    ],
                }
                for index, clause_id in enumerate(clause_ids, start=1)
            ],
            "completeness": completeness_items(),
        }
    )


class WholeContractReviewServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_request_contains_all_clauses_and_unique_sources(self) -> None:
        first = make_clause_result("clause-0001")
        second = make_clause_result(
            "clause-0002",
            text="Employee receives one rest day per week.",
        )
        raw = whole_review_json(["clause-0001", "clause-0002"])
        gateway = FakeGateway([raw])

        result = await WholeContractReviewService(gateway).review(
            make_preparation([first, second])
        )

        self.assertEqual(len(gateway.requests), 1)
        self.assertEqual(result.status, ReviewRunStatus.READY)
        self.assertEqual(len(result.clause_reviews), 2)
        self.assertEqual(len(result.unique_legal_sources), 1)
        self.assertEqual(len(result.evidence_mappings), 2)
        self.assertEqual(
            {mapping.clause_id for mapping in result.evidence_mappings},
            {"clause-0001", "clause-0002"},
        )
        prompt = gateway.requests[0].messages[1].content
        self.assertIn("clause-0001", prompt)
        self.assertIn("clause-0002", prompt)
        unique_source_section = prompt.split("UNIQUE_LEGAL_SOURCES", 1)[1].split(
            "EVIDENCE_MAPPINGS", 1
        )[0]
        self.assertEqual(unique_source_section.count("law:overtime"), 1)
        schema = gateway.requests[0].response_format["json_schema"]["schema"]
        assessment_properties = schema["$defs"][
            "GeneratedWholeContractAssessment"
        ]["properties"]
        self.assertNotIn("evidence_gap", assessment_properties)
        self.assertNotIn("suggested_revision", assessment_properties)
        citation_source_schema = schema["$defs"]["GeneratedLegalCitation"][
            "properties"
        ]["source_id"]
        self.assertEqual(citation_source_schema["enum"], ["law:overtime"])
        self.assertIn("opaque identifier", gateway.requests[0].messages[0].content)
        completeness_schema = schema["properties"]["completeness"]
        self.assertEqual(completeness_schema["minItems"], 8)
        self.assertEqual(completeness_schema["maxItems"], 8)
        completeness_properties = schema["$defs"][
            "ContractCompletenessItem"
        ]["properties"]
        self.assertEqual(
            set(completeness_properties),
            {"category", "label", "status", "summary"},
        )
        self.assertEqual(len(result.completeness), 8)
        self.assertEqual(
            [item.category.value for item in result.completeness],
            list(COMPLETENESS_CATEGORY_LABELS),
        )
        self.assertIn("parties_information", gateway.requests[0].messages[0].content)
        self.assertTrue(result.llm_debug.accepted)
        self.assertEqual(result.llm_debug.response.content, raw)
        self.assertEqual(result.llm_debug.response.request_id, "test-request-1")

    async def test_all_completeness_statuses_are_accepted(self) -> None:
        payload = json.loads(whole_review_json(["clause-0001"]))
        statuses = [
            "complete",
            "partial",
            "missing",
            "not_applicable",
            "uncertain",
        ]
        for index, status in enumerate(statuses):
            payload["completeness"][index]["status"] = status
            payload["completeness"][index]["summary"] = (
                f"summary for {status}"
            )
        gateway = FakeGateway([json.dumps(payload)])

        result = await WholeContractReviewService(gateway).review(
            make_preparation([make_clause_result("clause-0001")])
        )

        self.assertEqual(result.status, ReviewRunStatus.READY)
        self.assertEqual(
            [item.status.value for item in result.completeness[:5]],
            statuses,
        )

    async def test_invalid_completeness_is_rejected(self) -> None:
        base_payload = json.loads(whole_review_json(["clause-0001"]))
        invalid_payloads: dict[str, dict[str, object]] = {}

        missing = json.loads(json.dumps(base_payload))
        missing["completeness"].pop()
        invalid_payloads["missing category"] = missing

        duplicate = json.loads(json.dumps(base_payload))
        duplicate["completeness"][-1] = duplicate["completeness"][0]
        invalid_payloads["duplicate category"] = duplicate

        reordered = json.loads(json.dumps(base_payload))
        reordered["completeness"][0], reordered["completeness"][1] = (
            reordered["completeness"][1],
            reordered["completeness"][0],
        )
        invalid_payloads["wrong order"] = reordered

        unknown = json.loads(json.dumps(base_payload))
        unknown["completeness"][0]["category"] = "unknown"
        invalid_payloads["unknown category"] = unknown

        blank = json.loads(json.dumps(base_payload))
        blank["completeness"][0]["summary"] = "   "
        invalid_payloads["blank summary"] = blank

        wrong_label = json.loads(json.dumps(base_payload))
        wrong_label["completeness"][0]["label"] = "Wrong label"
        invalid_payloads["wrong fixed label"] = wrong_label

        invalid_status = json.loads(json.dumps(base_payload))
        invalid_status["completeness"][0]["status"] = "mostly_complete"
        invalid_payloads["invalid status"] = invalid_status

        extra = json.loads(json.dumps(base_payload))
        extra["completeness"][0]["evidence"] = []
        invalid_payloads["extra field"] = extra

        for case, payload in invalid_payloads.items():
            with self.subTest(case=case):
                gateway = FakeGateway([json.dumps(payload)])
                result = await WholeContractReviewService(gateway).review(
                    make_preparation([make_clause_result("clause-0001")])
                )

                self.assertEqual(result.status, ReviewRunStatus.FAILED)
                self.assertEqual(
                    result.llm_debug.error_code,
                    "invalid_model_output",
                )
                self.assertEqual(result.completeness, [])

    async def test_missing_clause_rejects_entire_response(self) -> None:
        gateway = FakeGateway([whole_review_json(["clause-0001"])])

        result = await WholeContractReviewService(gateway).review(
            make_preparation(
                [
                    make_clause_result("clause-0001"),
                    make_clause_result("clause-0002"),
                ]
            )
        )

        self.assertEqual(result.status, ReviewRunStatus.FAILED)
        self.assertEqual(
            result.llm_debug.error_code,
            "clause_coverage_mismatch",
        )
        self.assertEqual(
            {review.status for review in result.clause_reviews},
            {ClauseReviewStatus.REVIEW_FAILED},
        )

    async def test_unknown_citation_is_rejected_and_raw_response_is_kept(self) -> None:
        raw = whole_review_json(["clause-0001"], source_id="law:invented")
        gateway = FakeGateway([raw])

        with self.assertLogs("uvicorn.error", level="WARNING") as captured:
            result = await WholeContractReviewService(gateway).review(
                make_preparation([make_clause_result("clause-0001")])
            )

        self.assertEqual(result.llm_debug.error_code, "unknown_citation")
        self.assertEqual(result.llm_debug.response.content, raw)
        logged = json.loads(
            captured.output[0].split("whole_contract_validation=", 1)[1]
        )
        self.assertEqual(
            logged["invalid_citations"],
            [
                {
                    "clause_id": "clause-0001",
                    "assessment_id": "assessment-01",
                    "source_id": "law:invented",
                }
            ],
        )
        self.assertNotIn("Employee", captured.output[0])

    async def test_unknown_citation_only_fails_its_clause(self) -> None:
        payload = json.loads(
            whole_review_json(["clause-0001", "clause-0002"])
        )
        payload["clause_reviews"][1]["assessments"][0]["citations"][0][
            "source_id"
        ] = "law:invented"
        gateway = FakeGateway([json.dumps(payload)])

        with self.assertLogs("uvicorn.error", level="WARNING"):
            result = await WholeContractReviewService(gateway).review(
                make_preparation(
                    [
                        make_clause_result("clause-0001"),
                        make_clause_result("clause-0002"),
                    ]
                )
            )

        self.assertEqual(result.status, ReviewRunStatus.PARTIAL)
        self.assertEqual(
            result.clause_reviews[0].status,
            ClauseReviewStatus.NO_ISSUE_FOUND,
        )
        self.assertEqual(
            result.clause_reviews[1].status,
            ClauseReviewStatus.REVIEW_FAILED,
        )
        self.assertEqual(
            result.clause_reviews[1].error_code,
            "unknown_citation",
        )
        self.assertEqual(
            result.llm_debug.error_code,
            "partial_unknown_citation",
        )
        self.assertEqual(len(result.completeness), 8)

    async def test_unknown_supporting_clause_is_rejected(self) -> None:
        payload = json.loads(whole_review_json(["clause-0001"]))
        payload["clause_reviews"][0]["assessments"][0][
            "supporting_contract_clause_ids"
        ] = ["clause-0001", "clause-invented"]
        gateway = FakeGateway([json.dumps(payload)])

        result = await WholeContractReviewService(gateway).review(
            make_preparation([make_clause_result("clause-0001")])
        )

        self.assertEqual(
            result.llm_debug.error_code,
            "unknown_supporting_contract_clause",
        )

    async def test_clause_without_sources_is_still_sent_to_llm(self) -> None:
        no_source = make_clause_result("clause-0001", with_source=False)
        raw = json.dumps(
            {
                "clause_reviews": [
                    {
                        "clause_id": "clause-0001",
                        "applicable": True,
                        "applicability_reason": "มีข้อกำหนดการจ้างงาน",
                        "assessments": [
                            {
                                "assessment_id": "assessment-01",
                                "outcome": "needs_more_evidence",
                                "issue": "หลักฐานกฎหมายไม่เพียงพอ",
                                "reasoning": "ไม่มีแหล่งกฎหมายที่ค้นคืนได้",
                                "citations": [],
                                "confidence": "low",
                                "supporting_contract_clause_ids": [
                                    "clause-0001"
                                ],
                            }
                        ],
                    }
                ],
                "completeness": completeness_items(),
            }
        )
        gateway = FakeGateway([raw])

        result = await WholeContractReviewService(gateway).review(
            make_preparation([no_source])
        )

        self.assertEqual(len(gateway.requests), 1)
        self.assertEqual(
            result.clause_reviews[0].status,
            ClauseReviewStatus.NEEDS_MORE_EVIDENCE,
        )


    async def test_invalid_clause_does_not_discard_valid_clause_reviews(self) -> None:
        payload = json.loads(
            whole_review_json(["clause-0001", "clause-0002"])
        )
        payload["clause_reviews"][1]["assessments"][0]["citations"] = []
        gateway = FakeGateway([json.dumps(payload)])

        result = await WholeContractReviewService(gateway).review(
            make_preparation(
                [
                    make_clause_result("clause-0001"),
                    make_clause_result("clause-0002"),
                ]
            )
        )

        self.assertEqual(result.status, ReviewRunStatus.PARTIAL)
        self.assertEqual(
            result.clause_reviews[0].status,
            ClauseReviewStatus.NO_ISSUE_FOUND,
        )
        self.assertEqual(
            result.clause_reviews[1].status,
            ClauseReviewStatus.REVIEW_FAILED,
        )
        self.assertEqual(result.summary.assessment_count, 1)
        self.assertEqual(
            result.llm_debug.error_code,
            "partial_invalid_model_output",
        )
        self.assertEqual(
            result.llm_debug.validation_errors[0].location[:2],
            ["clause_reviews", 1],
        )



    def test_logs_model_provider_and_detailed_usage_without_content(self) -> None:
        response = LLMResponse(
            content="sensitive contract analysis",
            model="openai/gpt-5-mini-2025-08-07",
            provider="Azure",
            finish_reason="stop",
            request_id="generation-123",
            usage=LLMUsage(total_tokens=130),
            raw_response={
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 30,
                    "total_tokens": 130,
                    "cost": 0.001,
                    "completion_tokens_details": {
                        "reasoning_tokens": 12
                    },
                },
                "openrouter_metadata": {
                    "requested": "openai/gpt-5-mini",
                    "region": "BKK",
                },
            },
        )

        with self.assertLogs(
            "uvicorn.error",
            level="INFO",
        ) as captured:
            WholeContractReviewService._log_model_usage(response)

        logged = json.loads(
            captured.output[0].split("whole_contract_llm=", 1)[1]
        )
        self.assertEqual(logged["model"], response.model)
        self.assertEqual(logged["provider"], "Azure")
        self.assertEqual(logged["usage"]["total_tokens"], 130)
        self.assertEqual(
            logged["usage"]["completion_tokens_details"]["reasoning_tokens"],
            12,
        )
        self.assertEqual(
            logged["provider_metadata"]["requested"],
            "openai/gpt-5-mini",
        )
        self.assertNotIn(response.content, captured.output[0])


if __name__ == "__main__":
    unittest.main()

import json
import unittest

from app.llm.base import LLMGateway
from app.llm.errors import LLMTimeoutError
from app.llm.models import LLMRequest, LLMResponse, LLMUsage
from app.reviews.models import (
    BoundaryReason,
    ClauseRetrievalPreparation,
    ClauseRetrievalResult,
    ClauseQuery,
    ContractClause,
    PreparationStatus,
    PreparationSummary,
    QueryRetrievalResult,
    RetrievedLegalSource,
    RetrievalStatus,
)
from app.reviews.review_models import (
    ClauseReviewStatus,
    LegalAssessmentOutcome,
    ReviewRunStatus,
)
from app.reviews.reviewer import ClauseReviewService


CLAUSE_TEXT = "Employee must work overtime whenever the employer requires."


class FakeGateway(LLMGateway):
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResponse(
            content=outcome,
            model="test/model",
            provider="test-provider",
            finish_reason="stop",
            request_id="test-request-1",
            usage=LLMUsage(
                prompt_tokens=100,
                completion_tokens=30,
                total_tokens=130,
                cost=0.001,
            ),
        )


def make_clause_result(
    clause_id: str = "clause-0001",
    *,
    text: str = CLAUSE_TEXT,
    with_source: bool = True,
) -> ClauseRetrievalResult:
    clause = ContractClause(
        clause_id=clause_id,
        order=0,
        number="1",
        title="Overtime",
        text=text,
        block_indices=[1],
        section_path=["Overtime"],
        boundary_reason=BoundaryReason.NUMBERED_CLAUSE,
    )
    query = ClauseQuery(
        query_id=f"{clause_id}-q01",
        clause_id=clause_id,
        text=text,
        block_indices=[1],
    )
    source = RetrievedLegalSource(
        source_id="law:overtime",
        law_name="Labor Protection Act",
        section="Section 24",
        text="Overtime generally requires the employee's prior consent.",
        source_file="labor-law.jsonl",
        best_score=0.9,
        best_rank=1,
        matched_query_ids=[query.query_id],
    )
    return ClauseRetrievalResult(
        clause=clause,
        queries=[query],
        query_results=[
            QueryRetrievalResult(
                query_id=query.query_id,
                status=(
                    RetrievalStatus.READY
                    if with_source
                    else RetrievalStatus.NO_RESULTS
                ),
                source_ids=[source.source_id] if with_source else [],
            )
        ],
        sources=[source] if with_source else [],
        status=(
            RetrievalStatus.READY
            if with_source
            else RetrievalStatus.NO_RESULTS
        ),
    )


def make_preparation(
    clause_results: list[ClauseRetrievalResult],
) -> ClauseRetrievalPreparation:
    return ClauseRetrievalPreparation(
        document_id="document-1",
        filename="contract.docx",
        status=PreparationStatus.READY,
        clause_results=clause_results,
        summary=PreparationSummary(
            block_count=len(clause_results),
            clause_count=len(clause_results),
            query_count=len(clause_results),
            retrieved_source_count=sum(
                len(result.sources) for result in clause_results
            ),
            ready_clause_count=sum(
                result.status == RetrievalStatus.READY
                for result in clause_results
            ),
            partial_clause_count=0,
            no_results_clause_count=sum(
                result.status == RetrievalStatus.NO_RESULTS
                for result in clause_results
            ),
            failed_clause_count=0,
        ),
    )


def assessment(
    assessment_id: str,
    outcome: str,
    *,
    source_id: str = "law:overtime",
) -> dict[str, object]:
    if outcome == "potential_violation":
        evidence_gap = None
        suggested_revision = "Require prior consent for each overtime period."
        citations = [{"source_id": source_id}]
    elif outcome == "compliant":
        evidence_gap = None
        suggested_revision = None
        citations = [{"source_id": source_id}]
    else:
        evidence_gap = "The retrieved evidence does not cover the exception."
        suggested_revision = None
        citations = [{"source_id": source_id}]

    return {
        "assessment_id": assessment_id,
        "outcome": outcome,
        "issue": f"{outcome} issue",
        "reasoning": f"Reasoning for {outcome}.",
        "citations": citations,
        "evidence_gap": evidence_gap,
        "suggested_revision": suggested_revision,
        "confidence": "high",
    }


def review_json(
    assessments: list[dict[str, object]],
    *,
    clause_id: str = "clause-0001",
    applicable: bool = True,
) -> str:
    return json.dumps(
        {
            "clause_id": clause_id,
            "applicable": applicable,
            "applicability_reason": (
                "The clause contains an employment term."
                if applicable
                else "The text has no substantive employment term."
            ),
            "assessments": assessments,
        }
    )


class ClauseReviewServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_assessment_is_enriched_with_retrieved_citation(
        self,
    ) -> None:
        raw = review_json([assessment("assessment-01", "potential_violation")])
        diagnostics = []
        gateway = FakeGateway([raw])
        service = ClauseReviewService(
            gateway,
            diagnostic_callback=diagnostics.append,
        )

        result = await service.review(make_preparation([make_clause_result()]))

        self.assertEqual(result.status, ReviewRunStatus.READY)
        review = result.clause_reviews[0]
        self.assertEqual(review.status, ClauseReviewStatus.POTENTIAL_VIOLATION)
        self.assertEqual(
            review.assessments[0].citations[0].section,
            "Section 24",
        )
        self.assertEqual(
            review.assessments[0].citations[0].source_id,
            "law:overtime",
        )
        self.assertEqual(review.clause.clause_id, "clause-0001")
        self.assertNotIn(
            "contract_excerpt",
            review.assessments[0].model_dump(),
        )
        self.assertEqual(review.usage.total_tokens, 130)
        self.assertEqual(result.summary.assessment_count, 1)
        self.assertEqual(
            result.summary.potential_violation_assessment_count,
            1,
        )
        self.assertEqual(len(diagnostics), 1)
        self.assertTrue(diagnostics[0].accepted)
        self.assertEqual(diagnostics[0].raw_model_output, raw)
        self.assertEqual(diagnostics[0].finish_reason, "stop")
        self.assertEqual(diagnostics[0].request_id, "test-request-1")
        self.assertEqual(
            gateway.requests[0].response_format["type"],
            "json_schema",
        )
        assessment_schema = gateway.requests[0].response_format["json_schema"][
            "schema"
        ]["$defs"]["GeneratedLegalAssessment"]
        self.assertNotIn("contract_excerpt", assessment_schema["properties"])
        self.assertIn("LEGAL_SOURCES", gateway.requests[0].messages[1].content)

    async def test_compliant_assessment_is_retained_for_no_issue_clause(
        self,
    ) -> None:
        gateway = FakeGateway(
            [review_json([assessment("assessment-01", "compliant")])]
        )

        result = await ClauseReviewService(gateway).review(
            make_preparation([make_clause_result()])
        )

        review = result.clause_reviews[0]
        self.assertEqual(review.status, ClauseReviewStatus.NO_ISSUE_FOUND)
        self.assertEqual(len(review.assessments), 1)
        self.assertEqual(
            review.assessments[0].outcome,
            LegalAssessmentOutcome.COMPLIANT,
        )
        self.assertEqual(result.summary.compliant_assessment_count, 1)

    async def test_mixed_assessments_derive_needs_more_evidence_status(
        self,
    ) -> None:
        gateway = FakeGateway(
            [
                review_json(
                    [
                        assessment("assessment-01", "compliant"),
                        assessment("assessment-02", "needs_more_evidence"),
                    ]
                )
            ]
        )

        result = await ClauseReviewService(gateway).review(
            make_preparation([make_clause_result()])
        )

        review = result.clause_reviews[0]
        self.assertEqual(
            review.status,
            ClauseReviewStatus.NEEDS_MORE_EVIDENCE,
        )
        self.assertEqual(len(review.assessments), 2)
        self.assertIn("does not cover", review.evidence_gap)

    async def test_violation_has_priority_in_derived_clause_status(self) -> None:
        gateway = FakeGateway(
            [
                review_json(
                    [
                        assessment("assessment-01", "needs_more_evidence"),
                        assessment("assessment-02", "potential_violation"),
                    ]
                )
            ]
        )

        result = await ClauseReviewService(gateway).review(
            make_preparation([make_clause_result()])
        )

        self.assertEqual(
            result.clause_reviews[0].status,
            ClauseReviewStatus.POTENTIAL_VIOLATION,
        )
        self.assertEqual(len(result.clause_reviews[0].assessments), 2)

    async def test_not_applicable_clause_has_no_assessments(self) -> None:
        gateway = FakeGateway([review_json([], applicable=False)])

        result = await ClauseReviewService(gateway).review(
            make_preparation([make_clause_result()])
        )

        review = result.clause_reviews[0]
        self.assertEqual(review.status, ClauseReviewStatus.NOT_APPLICABLE)
        self.assertEqual(review.assessments, [])

    async def test_unknown_citation_rejects_complete_clause_response(self) -> None:
        gateway = FakeGateway(
            [
                review_json(
                    [
                        assessment(
                            "assessment-01",
                            "potential_violation",
                            source_id="law:invented",
                        )
                    ]
                )
            ]
        )

        result = await ClauseReviewService(gateway).review(
            make_preparation([make_clause_result()])
        )

        review = result.clause_reviews[0]
        self.assertEqual(review.status, ClauseReviewStatus.REVIEW_FAILED)
        self.assertEqual(review.error_code, "unknown_citation")
        self.assertEqual(review.assessments, [])
        self.assertEqual(result.status, ReviewRunStatus.FAILED)

    async def test_no_retrieved_source_skips_language_model(self) -> None:
        gateway = FakeGateway([])
        service = ClauseReviewService(gateway)

        result = await service.review(
            make_preparation([make_clause_result(with_source=False)])
        )

        review = result.clause_reviews[0]
        self.assertEqual(
            review.status,
            ClauseReviewStatus.NEEDS_MORE_EVIDENCE,
        )
        self.assertIsNotNone(review.evidence_gap)
        self.assertEqual(review.assessments, [])
        self.assertEqual(gateway.requests, [])
        self.assertEqual(result.status, ReviewRunStatus.READY)

    async def test_invalid_json_is_isolated_from_other_clause(self) -> None:
        second_clause = make_clause_result(
            "clause-0002",
            text="Second contract clause.",
        )
        gateway = FakeGateway(
            [
                "not-json",
                review_json(
                    [
                        assessment(
                            "assessment-01",
                            "compliant",
                        )
                    ],
                    clause_id="clause-0002",
                ),
            ]
        )
        service = ClauseReviewService(gateway, max_concurrency=1)

        result = await service.review(
            make_preparation([make_clause_result(), second_clause])
        )

        self.assertEqual(result.status, ReviewRunStatus.PARTIAL)
        self.assertEqual(
            [review.status for review in result.clause_reviews],
            [
                ClauseReviewStatus.REVIEW_FAILED,
                ClauseReviewStatus.NO_ISSUE_FOUND,
            ],
        )
        self.assertEqual(result.summary.review_failed_count, 1)

    async def test_gateway_error_uses_stable_error_code(self) -> None:
        gateway = FakeGateway(
            [
                LLMTimeoutError(
                    "private provider detail",
                    code="llm_timeout",
                    retryable=True,
                )
            ]
        )

        result = await ClauseReviewService(gateway).review(
            make_preparation([make_clause_result()])
        )

        review = result.clause_reviews[0]
        self.assertEqual(review.status, ClauseReviewStatus.REVIEW_FAILED)
        self.assertEqual(review.error_code, "llm_timeout")

    async def test_invalid_assessment_payload_is_rejected(self) -> None:
        diagnostics = []
        invalid = assessment("assessment-01", "potential_violation")
        invalid["suggested_revision"] = None
        gateway = FakeGateway([review_json([invalid])])
        service = ClauseReviewService(
            gateway,
            diagnostic_callback=diagnostics.append,
        )

        result = await service.review(make_preparation([make_clause_result()]))

        review = result.clause_reviews[0]
        self.assertEqual(review.status, ClauseReviewStatus.REVIEW_FAILED)
        self.assertEqual(review.error_code, "invalid_model_output")
        self.assertEqual(len(diagnostics), 1)
        self.assertFalse(diagnostics[0].accepted)
        self.assertEqual(
            diagnostics[0].validation_errors[0].error_type,
            "value_error",
        )

    async def test_invalid_json_diagnostic_keeps_exact_raw_response(self) -> None:
        diagnostics = []
        raw_response = '{"clause_id": "clause-0001"'
        gateway = FakeGateway([raw_response])
        service = ClauseReviewService(
            gateway,
            diagnostic_callback=diagnostics.append,
        )

        result = await service.review(make_preparation([make_clause_result()]))

        self.assertEqual(
            result.clause_reviews[0].status,
            ClauseReviewStatus.REVIEW_FAILED,
        )
        self.assertEqual(diagnostics[0].raw_model_output, raw_response)
        self.assertEqual(
            diagnostics[0].validation_errors[0].error_type,
            "json_invalid",
        )


if __name__ == "__main__":
    unittest.main()

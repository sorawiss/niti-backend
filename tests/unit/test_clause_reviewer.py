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
from app.reviews.review_models import ClauseReviewStatus, ReviewRunStatus
from app.reviews.reviewer import ClauseReviewService


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
    text: str = "ลูกจ้างตกลงทำงานล่วงเวลาตามที่นายจ้างกำหนด",
    with_source: bool = True,
) -> ClauseRetrievalResult:
    clause = ContractClause(
        clause_id=clause_id,
        order=0,
        number="1",
        title="การทำงานล่วงเวลา",
        text=text,
        block_indices=[1],
        section_path=["การทำงานล่วงเวลา"],
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
        law_name="พระราชบัญญัติคุ้มครองแรงงาน พ.ศ. 2541",
        section="มาตรา 24",
        text="การทำงานล่วงเวลาต้องได้รับความยินยอมจากลูกจ้างก่อนเป็นคราว ๆ ไป",
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
            RetrievalStatus.READY if with_source else RetrievalStatus.NO_RESULTS
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


def potential_violation_json(
    clause_id: str = "clause-0001",
    *,
    source_id: str = "law:overtime",
    excerpt: str = "ลูกจ้างตกลงทำงานล่วงเวลาตามที่นายจ้างกำหนด",
) -> str:
    return json.dumps(
        {
            "clause_id": clause_id,
            "status": "potential_violation",
            "findings": [
                {
                    "finding_id": "finding-01",
                    "contract_excerpt": excerpt,
                    "issue": "ขาดเงื่อนไขเรื่องความยินยอม",
                    "reasoning": "หลักฐานกำหนดให้ได้รับความยินยอมก่อน",
                    "citations": [{"source_id": source_id}],
                    "suggested_revision": (
                        "การทำงานล่วงเวลาต้องได้รับความยินยอมก่อนเป็นคราว ๆ ไป"
                    ),
                    "confidence": "high",
                }
            ],
            "evidence_gap": None,
        },
        ensure_ascii=False,
    )


class ClauseReviewServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_finding_is_enriched_with_retrieved_citation(self) -> None:
        gateway = FakeGateway([potential_violation_json()])
        service = ClauseReviewService(gateway)

        result = await service.review(make_preparation([make_clause_result()]))

        self.assertEqual(result.status, ReviewRunStatus.READY)
        review = result.clause_reviews[0]
        self.assertEqual(review.status, ClauseReviewStatus.POTENTIAL_VIOLATION)
        self.assertEqual(review.findings[0].citations[0].section, "มาตรา 24")
        self.assertEqual(review.findings[0].citations[0].source_id, "law:overtime")
        self.assertEqual(review.usage.total_tokens, 130)
        self.assertEqual(result.summary.finding_count, 1)
        self.assertEqual(
            gateway.requests[0].response_format["type"],
            "json_schema",
        )
        self.assertIn("LEGAL_SOURCES", gateway.requests[0].messages[1].content)

    async def test_unknown_citation_rejects_complete_clause_response(self) -> None:
        gateway = FakeGateway(
            [potential_violation_json(source_id="law:invented")]
        )
        service = ClauseReviewService(gateway)

        result = await service.review(make_preparation([make_clause_result()]))

        review = result.clause_reviews[0]
        self.assertEqual(review.status, ClauseReviewStatus.REVIEW_FAILED)
        self.assertEqual(review.error_code, "unknown_citation")
        self.assertEqual(review.findings, [])
        self.assertEqual(result.status, ReviewRunStatus.FAILED)

    async def test_non_verbatim_excerpt_rejects_complete_response(self) -> None:
        gateway = FakeGateway(
            [potential_violation_json(excerpt="ข้อความที่ไม่มีในสัญญา")]
        )
        service = ClauseReviewService(gateway)

        result = await service.review(make_preparation([make_clause_result()]))

        review = result.clause_reviews[0]
        self.assertEqual(review.status, ClauseReviewStatus.REVIEW_FAILED)
        self.assertEqual(review.error_code, "contract_excerpt_mismatch")
        self.assertEqual(review.findings, [])

    async def test_no_retrieved_source_skips_language_model(self) -> None:
        gateway = FakeGateway([])
        service = ClauseReviewService(gateway)

        result = await service.review(
            make_preparation([make_clause_result(with_source=False)])
        )

        review = result.clause_reviews[0]
        self.assertEqual(review.status, ClauseReviewStatus.NEEDS_MORE_EVIDENCE)
        self.assertIsNotNone(review.evidence_gap)
        self.assertEqual(gateway.requests, [])
        self.assertEqual(result.status, ReviewRunStatus.READY)

    async def test_invalid_json_is_isolated_from_other_clause(self) -> None:
        second_clause = make_clause_result(
            "clause-0002",
            text="ข้อสัญญาที่สอง",
        )
        gateway = FakeGateway(
            [
                "not-json",
                json.dumps(
                    {
                        "clause_id": "clause-0002",
                        "status": "no_issue_found",
                        "findings": [],
                        "evidence_gap": None,
                    }
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
        service = ClauseReviewService(gateway)

        result = await service.review(make_preparation([make_clause_result()]))

        review = result.clause_reviews[0]
        self.assertEqual(review.status, ClauseReviewStatus.REVIEW_FAILED)
        self.assertEqual(review.error_code, "llm_timeout")

    async def test_invalid_status_payload_is_rejected(self) -> None:
        gateway = FakeGateway(
            [
                json.dumps(
                    {
                        "clause_id": "clause-0001",
                        "status": "potential_violation",
                        "findings": [],
                        "evidence_gap": None,
                    }
                )
            ]
        )
        service = ClauseReviewService(gateway)

        result = await service.review(make_preparation([make_clause_result()]))

        review = result.clause_reviews[0]
        self.assertEqual(review.status, ClauseReviewStatus.REVIEW_FAILED)
        self.assertEqual(review.error_code, "invalid_model_output")


if __name__ == "__main__":
    unittest.main()

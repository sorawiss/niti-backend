import asyncio
from typing import Any

from pydantic import ValidationError

from app.llm.base import LLMGateway
from app.llm.errors import LLMGatewayError
from app.llm.models import LLMRequest, LLMResponse, LLMUsage
from app.reviews.models import ClauseRetrievalPreparation, ClauseRetrievalResult
from app.reviews.review_models import (
    ClauseReviewResult,
    ClauseReviewStatus,
    ContractReviewResult,
    GeneratedClauseReview,
    LegalCitation,
    LegalFinding,
    ReviewRunStatus,
    ReviewSummary,
)
from app.reviews.review_prompts import build_clause_review_messages


class ClauseReviewService:
    """Review retrieved clause evidence and expose only validated findings."""

    def __init__(
        self,
        llm_gateway: LLMGateway,
        *,
        max_concurrency: int = 1,
        max_output_tokens: int = 2000,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        self._llm_gateway = llm_gateway
        self._max_concurrency = max_concurrency
        self._max_output_tokens = max_output_tokens

    async def review(
        self,
        preparation: ClauseRetrievalPreparation,
    ) -> ContractReviewResult:
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def review_one(
            clause_result: ClauseRetrievalResult,
        ) -> ClauseReviewResult:
            if not clause_result.sources:
                return ClauseReviewResult(
                    clause=clause_result.clause,
                    retrieval_status=clause_result.status,
                    status=ClauseReviewStatus.NEEDS_MORE_EVIDENCE,
                    evidence_gap=(
                        "ไม่พบหลักฐานกฎหมายจากการค้นคืนสำหรับข้อสัญญานี้"
                    ),
                )
            async with semaphore:
                return await self._review_with_llm(clause_result)

        clause_reviews = list(
            await asyncio.gather(
                *(review_one(result) for result in preparation.clause_results)
            )
        )
        return ContractReviewResult(
            document_id=preparation.document_id,
            filename=preparation.filename,
            status=self._run_status(clause_reviews),
            clause_reviews=clause_reviews,
            summary=self._summary(clause_reviews),
        )

    async def _review_with_llm(
        self,
        clause_result: ClauseRetrievalResult,
    ) -> ClauseReviewResult:
        try:
            response = await self._llm_gateway.generate(
                LLMRequest(
                    messages=build_clause_review_messages(clause_result),
                    max_output_tokens=self._max_output_tokens,
                    response_format=self._response_format(),
                )
            )
        except LLMGatewayError as error:
            return self._failed_result(clause_result, error.code)
        except Exception:
            return self._failed_result(clause_result, "llm_unexpected_error")

        try:
            generated = GeneratedClauseReview.model_validate_json(response.content)
        except ValidationError:
            return self._failed_result(
                clause_result,
                "invalid_model_output",
                response=response,
            )

        validation_error = self._validate_provenance(generated, clause_result)
        if validation_error is not None:
            return self._failed_result(
                clause_result,
                validation_error,
                response=response,
            )

        sources_by_id = {
            source.source_id: source for source in clause_result.sources
        }
        findings = [
            LegalFinding(
                finding_id=finding.finding_id,
                contract_excerpt=finding.contract_excerpt,
                issue=finding.issue,
                reasoning=finding.reasoning,
                citations=[
                    LegalCitation(
                        source_id=source.source_id,
                        law_name=source.law_name,
                        section=source.section,
                        text=source.text,
                        source_file=source.source_file,
                    )
                    for source_id in dict.fromkeys(
                        citation.source_id for citation in finding.citations
                    )
                    for source in [sources_by_id[source_id]]
                ],
                suggested_revision=finding.suggested_revision,
                confidence=finding.confidence,
            )
            for finding in generated.findings
        ]
        return ClauseReviewResult(
            clause=clause_result.clause,
            retrieval_status=clause_result.status,
            status=ClauseReviewStatus(generated.status.value),
            findings=findings,
            evidence_gap=generated.evidence_gap,
            model=response.model,
            provider=response.provider,
            usage=response.usage,
        )

    @staticmethod
    def _validate_provenance(
        generated: GeneratedClauseReview,
        clause_result: ClauseRetrievalResult,
    ) -> str | None:
        if generated.clause_id != clause_result.clause.clause_id:
            return "clause_id_mismatch"

        finding_ids = [finding.finding_id for finding in generated.findings]
        if len(finding_ids) != len(set(finding_ids)):
            return "duplicate_finding_id"

        valid_source_ids = {source.source_id for source in clause_result.sources}
        for finding in generated.findings:
            if finding.contract_excerpt not in clause_result.clause.text:
                return "contract_excerpt_mismatch"
            if any(
                citation.source_id not in valid_source_ids
                for citation in finding.citations
            ):
                return "unknown_citation"
        return None

    @staticmethod
    def _failed_result(
        clause_result: ClauseRetrievalResult,
        error_code: str,
        *,
        response: LLMResponse | None = None,
    ) -> ClauseReviewResult:
        return ClauseReviewResult(
            clause=clause_result.clause,
            retrieval_status=clause_result.status,
            status=ClauseReviewStatus.REVIEW_FAILED,
            error_code=error_code,
            model=response.model if response else None,
            provider=response.provider if response else None,
            usage=response.usage if response else LLMUsage(),
        )

    @staticmethod
    def _response_format() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "clause_legal_review",
                "strict": True,
                "schema": GeneratedClauseReview.model_json_schema(),
            },
        }

    @staticmethod
    def _run_status(clause_reviews: list[ClauseReviewResult]) -> ReviewRunStatus:
        if not clause_reviews:
            return ReviewRunStatus.FAILED
        failed_count = sum(
            review.status == ClauseReviewStatus.REVIEW_FAILED
            for review in clause_reviews
        )
        if failed_count == 0:
            return ReviewRunStatus.READY
        if failed_count == len(clause_reviews):
            return ReviewRunStatus.FAILED
        return ReviewRunStatus.PARTIAL

    @staticmethod
    def _summary(clause_reviews: list[ClauseReviewResult]) -> ReviewSummary:
        counts = {
            status: sum(review.status == status for review in clause_reviews)
            for status in ClauseReviewStatus
        }
        return ReviewSummary(
            clause_count=len(clause_reviews),
            finding_count=sum(len(review.findings) for review in clause_reviews),
            no_issue_found_count=counts[ClauseReviewStatus.NO_ISSUE_FOUND],
            potential_violation_count=counts[
                ClauseReviewStatus.POTENTIAL_VIOLATION
            ],
            needs_more_evidence_count=counts[
                ClauseReviewStatus.NEEDS_MORE_EVIDENCE
            ],
            not_applicable_count=counts[ClauseReviewStatus.NOT_APPLICABLE],
            review_failed_count=counts[ClauseReviewStatus.REVIEW_FAILED],
        )

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from app.llm.base import LLMGateway
from app.llm.errors import LLMGatewayError
from app.llm.models import LLMRequest, LLMResponse, LLMUsage
from app.reviews.models import ClauseRetrievalPreparation, ClauseRetrievalResult
from app.reviews.review_models import (
    ClauseReviewDiagnostic,
    ClauseReviewResult,
    ClauseReviewStatus,
    ContractReviewResult,
    GeneratedClauseReview,
    LegalAssessment,
    LegalAssessmentOutcome,
    LegalCitation,
    ReviewRunStatus,
    ReviewSummary,
    ReviewValidationIssue,
)
from app.reviews.review_prompts import build_clause_review_messages


class ClauseReviewService:
    """Review retrieved clause evidence and expose validated assessments."""

    def __init__(
        self,
        llm_gateway: LLMGateway,
        *,
        max_concurrency: int = 1,
        max_output_tokens: int = 2000,
        diagnostic_callback: Callable[[ClauseReviewDiagnostic], None] | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        self._llm_gateway = llm_gateway
        self._max_concurrency = max_concurrency
        self._max_output_tokens = max_output_tokens
        self._diagnostic_callback = diagnostic_callback

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
                    applicability_reason=(
                        "ยังประเมินการใช้กฎหมายกับข้อนี้ไม่ได้เพราะไม่พบหลักฐาน"
                    ),
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
            self._emit_diagnostic(
                ClauseReviewDiagnostic(
                    clause_id=clause_result.clause.clause_id,
                    accepted=False,
                    error_code=error.code,
                    exception_type=type(error).__name__,
                )
            )
            return self._failed_result(clause_result, error.code)
        except Exception as error:
            self._emit_diagnostic(
                ClauseReviewDiagnostic(
                    clause_id=clause_result.clause.clause_id,
                    accepted=False,
                    error_code="llm_unexpected_error",
                    exception_type=type(error).__name__,
                )
            )
            return self._failed_result(clause_result, "llm_unexpected_error")

        try:
            generated = GeneratedClauseReview.model_validate_json(response.content)
        except ValidationError as error:
            self._emit_diagnostic(
                self._response_diagnostic(
                    clause_result,
                    response,
                    accepted=False,
                    error_code="invalid_model_output",
                    validation_errors=[
                        ReviewValidationIssue(
                            error_type=str(item["type"]),
                            location=list(item["loc"]),
                            message=str(item["msg"]),
                        )
                        for item in error.errors(
                            include_url=False,
                            include_context=False,
                            include_input=False,
                        )
                    ],
                )
            )
            return self._failed_result(
                clause_result,
                "invalid_model_output",
                response=response,
            )

        validation_error = self._validate_provenance(generated, clause_result)
        if validation_error is not None:
            self._emit_diagnostic(
                self._response_diagnostic(
                    clause_result,
                    response,
                    accepted=False,
                    error_code=validation_error,
                    validation_errors=[
                        ReviewValidationIssue(
                            error_type=validation_error,
                            location=["provenance"],
                            message=self._provenance_error_message(validation_error),
                        )
                    ],
                )
            )
            return self._failed_result(
                clause_result,
                validation_error,
                response=response,
            )

        sources_by_id = {
            source.source_id: source for source in clause_result.sources
        }
        assessments = [
            LegalAssessment(
                assessment_id=assessment.assessment_id,
                outcome=assessment.outcome,
                issue=assessment.issue,
                reasoning=assessment.reasoning,
                citations=[
                    LegalCitation(
                        source_id=source.source_id,
                        law_name=source.law_name,
                        section=source.section,
                        text=source.text,
                        source_file=source.source_file,
                    )
                    for source_id in dict.fromkeys(
                        citation.source_id for citation in assessment.citations
                    )
                    for source in [sources_by_id[source_id]]
                ],
                evidence_gap=assessment.evidence_gap,
                suggested_revision=assessment.suggested_revision,
                confidence=assessment.confidence,
            )
            for assessment in generated.assessments
        ]
        evidence_gaps = list(
            dict.fromkeys(
                assessment.evidence_gap
                for assessment in assessments
                if assessment.evidence_gap is not None
            )
        )
        self._emit_diagnostic(
            self._response_diagnostic(
                clause_result,
                response,
                accepted=True,
            )
        )
        return ClauseReviewResult(
            clause=clause_result.clause,
            retrieval_status=clause_result.status,
            status=self._derive_status(generated),
            applicability_reason=generated.applicability_reason,
            assessments=assessments,
            evidence_gap="\n".join(evidence_gaps) or None,
            model=response.model,
            provider=response.provider,
            usage=response.usage,
        )

    @staticmethod
    def _derive_status(generated: GeneratedClauseReview) -> ClauseReviewStatus:
        if not generated.applicable:
            return ClauseReviewStatus.NOT_APPLICABLE

        outcomes = {
            assessment.outcome for assessment in generated.assessments
        }
        if LegalAssessmentOutcome.POTENTIAL_VIOLATION in outcomes:
            return ClauseReviewStatus.POTENTIAL_VIOLATION
        if LegalAssessmentOutcome.NEEDS_MORE_EVIDENCE in outcomes:
            return ClauseReviewStatus.NEEDS_MORE_EVIDENCE
        return ClauseReviewStatus.NO_ISSUE_FOUND

    @staticmethod
    def _validate_provenance(
        generated: GeneratedClauseReview,
        clause_result: ClauseRetrievalResult,
    ) -> str | None:
        if generated.clause_id != clause_result.clause.clause_id:
            return "clause_id_mismatch"

        assessment_ids = [
            assessment.assessment_id for assessment in generated.assessments
        ]
        if len(assessment_ids) != len(set(assessment_ids)):
            return "duplicate_assessment_id"

        valid_source_ids = {source.source_id for source in clause_result.sources}
        for assessment in generated.assessments:

            if any(
                citation.source_id not in valid_source_ids
                for citation in assessment.citations
            ):
                return "unknown_citation"
        return None

    @staticmethod
    def _provenance_error_message(error_code: str) -> str:
        messages = {
            "clause_id_mismatch": (
                "Returned clause_id does not match the reviewed clause"
            ),
            "duplicate_assessment_id": (
                "Returned assessment_id values are not unique"
            ),

            "unknown_citation": (
                "Returned citation source_id was not present in "
                "retrieved evidence"
            ),
        }
        return messages.get(
            error_code,
            "Clause review provenance validation failed",
        )

    @staticmethod
    def _response_diagnostic(
        clause_result: ClauseRetrievalResult,
        response: LLMResponse,
        *,
        accepted: bool,
        error_code: str | None = None,
        validation_errors: list[ReviewValidationIssue] | None = None,
    ) -> ClauseReviewDiagnostic:
        return ClauseReviewDiagnostic(
            clause_id=clause_result.clause.clause_id,
            accepted=accepted,
            error_code=error_code,
            raw_model_output=response.content,
            validation_errors=validation_errors or [],
            finish_reason=response.finish_reason,
            request_id=response.request_id,
            model=response.model,
            provider=response.provider,
            usage=response.usage,
        )

    def _emit_diagnostic(self, diagnostic: ClauseReviewDiagnostic) -> None:
        if self._diagnostic_callback is not None:
            self._diagnostic_callback(diagnostic)

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
    def _run_status(
        clause_reviews: list[ClauseReviewResult],
    ) -> ReviewRunStatus:
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
        clause_counts = {
            status: sum(review.status == status for review in clause_reviews)
            for status in ClauseReviewStatus
        }
        assessments = [
            assessment
            for review in clause_reviews
            for assessment in review.assessments
        ]
        assessment_counts = {
            outcome: sum(
                assessment.outcome == outcome for assessment in assessments
            )
            for outcome in LegalAssessmentOutcome
        }
        return ReviewSummary(
            clause_count=len(clause_reviews),
            assessment_count=len(assessments),
            compliant_assessment_count=assessment_counts[
                LegalAssessmentOutcome.COMPLIANT
            ],
            potential_violation_assessment_count=assessment_counts[
                LegalAssessmentOutcome.POTENTIAL_VIOLATION
            ],
            needs_more_evidence_assessment_count=assessment_counts[
                LegalAssessmentOutcome.NEEDS_MORE_EVIDENCE
            ],
            no_issue_found_count=clause_counts[
                ClauseReviewStatus.NO_ISSUE_FOUND
            ],
            potential_violation_count=clause_counts[
                ClauseReviewStatus.POTENTIAL_VIOLATION
            ],
            needs_more_evidence_count=clause_counts[
                ClauseReviewStatus.NEEDS_MORE_EVIDENCE
            ],
            not_applicable_count=clause_counts[
                ClauseReviewStatus.NOT_APPLICABLE
            ],
            review_failed_count=clause_counts[
                ClauseReviewStatus.REVIEW_FAILED
            ],
        )

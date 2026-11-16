import json
import logging
from typing import Any

from pydantic import ValidationError

from app.llm.base import LLMGateway
from app.llm.errors import LLMGatewayError
from app.llm.models import LLMRequest, LLMResponse
from app.reviews.models import (
    ClauseRetrievalPreparation,
    ClauseRetrievalResult,
    RetrievedLegalSource,
    SourceMatch,
)
from app.reviews.review_models import (
    ClauseReviewStatus,
    LegalAssessmentOutcome,
    LegalCitation,
    ReviewRunStatus,
    ReviewSummary,
    ReviewValidationIssue,
)
from app.reviews.whole_contract_models import (
    ClauseQuerySourceMapping,
    GeneratedWholeContractClauseReview,
    GeneratedWholeContractReview,
    WholeContractClauseReviewResult,
    WholeContractLLMDebug,
    WholeContractLegalAssessment,
    WholeContractReviewResult,
)
from app.reviews.whole_contract_prompts import (
    build_whole_contract_review_messages,
)


logger = logging.getLogger("uvicorn.error")


class WholeContractReviewService:
    """Review every clause and all retrieved evidence in one LLM request."""

    def __init__(
        self,
        llm_gateway: LLMGateway,
        *,
        max_output_tokens: int = 16384,
    ) -> None:
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        self._llm_gateway = llm_gateway
        self._max_output_tokens = max_output_tokens

    async def review(
        self,
        preparation: ClauseRetrievalPreparation,
    ) -> WholeContractReviewResult:
        unique_sources = self._deduplicate_sources(preparation)
        evidence_mappings = self._build_evidence_mappings(preparation)
        try:
            response = await self._llm_gateway.generate(
                LLMRequest(
                    messages=build_whole_contract_review_messages(
                        preparation,
                        unique_sources,
                        evidence_mappings,
                    ),
                    max_output_tokens=self._max_output_tokens,
                    response_format=self._response_format(),
                )
            )
        except LLMGatewayError as error:
            return self._failed_result(
                preparation,
                unique_sources,
                evidence_mappings,
                error_code=error.code,
                exception_type=type(error).__name__,
            )
        except Exception as error:
            return self._failed_result(
                preparation,
                unique_sources,
                evidence_mappings,
                error_code="llm_unexpected_error",
                exception_type=type(error).__name__,
            )

        self._log_model_usage(response)

        try:
            generated = GeneratedWholeContractReview.model_validate_json(
                response.content
            )
        except ValidationError as error:
            return self._recover_clause_results(
                preparation,
                unique_sources,
                evidence_mappings,
                response,
                error,
            )

        provenance_error = self._validate_provenance(
            generated,
            preparation,
            unique_sources,
        )
        if provenance_error is not None:
            return self._failed_result(
                preparation,
                unique_sources,
                evidence_mappings,
                error_code=provenance_error,
                response=response,
                validation_errors=[
                    ReviewValidationIssue(
                        error_type=provenance_error,
                        location=["provenance"],
                        message=self._provenance_error_message(provenance_error),
                    )
                ],
            )

        sources_by_id = {source.source_id: source for source in unique_sources}
        generated_by_id = {
            review.clause_id: review for review in generated.clause_reviews
        }
        clause_reviews = [
            self._hydrate_clause_review(
                generated_by_id[result.clause.clause_id],
                result,
                sources_by_id,
            )
            for result in preparation.clause_results
        ]
        return WholeContractReviewResult(
            document_id=preparation.document_id,
            filename=preparation.filename,
            status=ReviewRunStatus.READY,
            clause_reviews=clause_reviews,
            summary=self._summary(clause_reviews),
            unique_legal_sources=unique_sources,
            evidence_mappings=evidence_mappings,
            llm_debug=WholeContractLLMDebug(
                accepted=True,
                response=response,
            ),
        )

    @staticmethod
    def _deduplicate_sources(
        preparation: ClauseRetrievalPreparation,
    ) -> list[RetrievedLegalSource]:
        sources_by_id: dict[str, RetrievedLegalSource] = {}
        for clause_result in preparation.clause_results:
            for source in clause_result.sources:
                existing = sources_by_id.get(source.source_id)
                if existing is None:
                    sources_by_id[source.source_id] = source.model_copy(deep=True)
                    continue

                existing.matched_query_ids = list(
                    dict.fromkeys(
                        [*existing.matched_query_ids, *source.matched_query_ids]
                    )
                )
                known_matches = {
                    (match.query_id, match.rank, match.score)
                    for match in existing.matches
                }
                existing.matches.extend(
                    SourceMatch(**match.model_dump())
                    for match in source.matches
                    if (match.query_id, match.rank, match.score) not in known_matches
                )
                existing.best_rank = min(existing.best_rank, source.best_rank)
                scores = [
                    score
                    for score in (existing.best_score, source.best_score)
                    if score is not None
                ]
                existing.best_score = max(scores) if scores else None
        return list(sources_by_id.values())

    @staticmethod
    def _build_evidence_mappings(
        preparation: ClauseRetrievalPreparation,
    ) -> list[ClauseQuerySourceMapping]:
        mappings: list[ClauseQuerySourceMapping] = []
        for clause_result in preparation.clause_results:
            queries_by_id = {query.query_id: query for query in clause_result.queries}
            for query_result in clause_result.query_results:
                query = queries_by_id[query_result.query_id]
                mappings.append(
                    ClauseQuerySourceMapping(
                        clause_id=clause_result.clause.clause_id,
                        query_id=query_result.query_id,
                        query_text=query.text,
                        retrieval_status=query_result.status,
                        source_ids=query_result.source_ids,
                        error_code=query_result.error_code,
                    )
                )
        return mappings

    @staticmethod
    def _validate_provenance(
        generated: GeneratedWholeContractReview,
        preparation: ClauseRetrievalPreparation,
        unique_sources: list[RetrievedLegalSource],
    ) -> str | None:
        expected_clause_ids = [
            result.clause.clause_id for result in preparation.clause_results
        ]
        returned_clause_ids = [
            review.clause_id for review in generated.clause_reviews
        ]
        if len(returned_clause_ids) != len(set(returned_clause_ids)):
            return "duplicate_clause_id"
        if set(returned_clause_ids) != set(expected_clause_ids):
            return "clause_coverage_mismatch"

        valid_clause_ids = set(expected_clause_ids)
        valid_source_ids = {source.source_id for source in unique_sources}
        for review in generated.clause_reviews:
            assessment_ids = [
                assessment.assessment_id for assessment in review.assessments
            ]
            if len(assessment_ids) != len(set(assessment_ids)):
                return "duplicate_assessment_id"
            for assessment in review.assessments:
                if any(
                    citation.source_id not in valid_source_ids
                    for citation in assessment.citations
                ):
                    return "unknown_citation"
                if any(
                    clause_id not in valid_clause_ids
                    for clause_id in assessment.supporting_contract_clause_ids
                ):
                    return "unknown_supporting_contract_clause"
                if (
                    review.clause_id
                    not in assessment.supporting_contract_clause_ids
                ):
                    return "missing_reviewed_clause_support"
        return None

    @staticmethod
    def _hydrate_clause_review(
        generated: GeneratedWholeContractClauseReview,
        clause_result: ClauseRetrievalResult,
        sources_by_id: dict[str, RetrievedLegalSource],
    ) -> WholeContractClauseReviewResult:
        assessments = [
            WholeContractLegalAssessment(
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
                confidence=assessment.confidence,
                supporting_contract_clause_ids=list(
                    dict.fromkeys(assessment.supporting_contract_clause_ids)
                ),
            )
            for assessment in generated.assessments
        ]
        return WholeContractClauseReviewResult(
            clause=clause_result.clause,
            retrieval_status=clause_result.status,
            status=WholeContractReviewService._derive_status(generated),
            applicability_reason=generated.applicability_reason,
            assessments=assessments,
        )

    @staticmethod
    def _derive_status(
        generated: GeneratedWholeContractClauseReview,
    ) -> ClauseReviewStatus:
        if not generated.applicable:
            return ClauseReviewStatus.NOT_APPLICABLE
        outcomes = {assessment.outcome for assessment in generated.assessments}
        if LegalAssessmentOutcome.POTENTIAL_VIOLATION in outcomes:
            return ClauseReviewStatus.POTENTIAL_VIOLATION
        if LegalAssessmentOutcome.NEEDS_MORE_EVIDENCE in outcomes:
            return ClauseReviewStatus.NEEDS_MORE_EVIDENCE
        return ClauseReviewStatus.NO_ISSUE_FOUND

    @staticmethod
    def _failed_result(
        preparation: ClauseRetrievalPreparation,
        unique_sources: list[RetrievedLegalSource],
        evidence_mappings: list[ClauseQuerySourceMapping],
        *,
        error_code: str,
        exception_type: str | None = None,
        response: LLMResponse | None = None,
        validation_errors: list[ReviewValidationIssue] | None = None,
    ) -> WholeContractReviewResult:
        clause_reviews = [
            WholeContractClauseReviewResult(
                clause=result.clause,
                retrieval_status=result.status,
                status=ClauseReviewStatus.REVIEW_FAILED,
                error_code=error_code,
            )
            for result in preparation.clause_results
        ]
        return WholeContractReviewResult(
            document_id=preparation.document_id,
            filename=preparation.filename,
            status=ReviewRunStatus.FAILED,
            clause_reviews=clause_reviews,
            summary=WholeContractReviewService._summary(clause_reviews),
            unique_legal_sources=unique_sources,
            evidence_mappings=evidence_mappings,
            llm_debug=WholeContractLLMDebug(
                accepted=False,
                error_code=error_code,
                exception_type=exception_type,
                response=response,
                validation_errors=validation_errors or [],
            ),
        )

    @staticmethod
    def _validation_issues(error: ValidationError) -> list[ReviewValidationIssue]:
        return [
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
        ]

    @staticmethod
    def _provenance_error_message(error_code: str) -> str:
        return {
            "duplicate_clause_id": "Returned clause_id values are not unique",
            "clause_coverage_mismatch": (
                "Returned clause IDs do not exactly cover every contract clause"
            ),
            "duplicate_assessment_id": (
                "Returned assessment_id values are not unique within a clause"
            ),
            "unknown_citation": (
                "Returned citation source_id was not present in retrieved evidence"
            ),
            "unknown_supporting_contract_clause": (
                "Returned supporting clause ID was not present in the contract"
            ),
            "missing_reviewed_clause_support": (
                "Assessment support does not include its reviewed clause_id"
            ),
        }.get(error_code, "Whole-contract provenance validation failed")

    @staticmethod
    def _summary(
        clause_reviews: list[WholeContractClauseReviewResult],
    ) -> ReviewSummary:
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
            review_failed_count=clause_counts[ClauseReviewStatus.REVIEW_FAILED],
        )

    @staticmethod
    def _response_format() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "whole_contract_legal_review",
                "strict": True,
                "schema": GeneratedWholeContractReview.model_json_schema(),
            },
        }
    def _recover_clause_results(
        self,
        preparation: ClauseRetrievalPreparation,
        unique_sources: list[RetrievedLegalSource],
        evidence_mappings: list[ClauseQuerySourceMapping],
        response: LLMResponse,
        whole_error: ValidationError,
    ) -> WholeContractReviewResult:
        """Keep valid clauses when one item violates the strict schema."""
        try:
            payload = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            return self._failed_result(
                preparation,
                unique_sources,
                evidence_mappings,
                error_code="invalid_model_output",
                response=response,
                validation_errors=self._validation_issues(whole_error),
            )

        raw_reviews = (
            payload.get("clause_reviews") if isinstance(payload, dict) else None
        )
        if (
            not isinstance(payload, dict)
            or set(payload) != {"clause_reviews"}
            or not isinstance(raw_reviews, list)
            or not all(isinstance(item, dict) for item in raw_reviews)
        ):
            return self._failed_result(
                preparation,
                unique_sources,
                evidence_mappings,
                error_code="invalid_model_output",
                response=response,
                validation_errors=self._validation_issues(whole_error),
            )

        expected_ids = [
            result.clause.clause_id for result in preparation.clause_results
        ]
        returned_ids = [item.get("clause_id") for item in raw_reviews]
        if (
            any(not isinstance(clause_id, str) for clause_id in returned_ids)
            or len(returned_ids) != len(set(returned_ids))
            or set(returned_ids) != set(expected_ids)
        ):
            return self._failed_result(
                preparation,
                unique_sources,
                evidence_mappings,
                error_code="clause_coverage_mismatch",
                response=response,
                validation_errors=self._validation_issues(whole_error),
            )

        raw_by_id = {item["clause_id"]: item for item in raw_reviews}
        index_by_id = {
            item["clause_id"]: index for index, item in enumerate(raw_reviews)
        }
        sources_by_id = {source.source_id: source for source in unique_sources}
        valid_clause_ids = set(expected_ids)
        clause_reviews: list[WholeContractClauseReviewResult] = []
        validation_errors: list[ReviewValidationIssue] = []

        for clause_result in preparation.clause_results:
            clause_id = clause_result.clause.clause_id
            index = index_by_id[clause_id]
            try:
                generated = GeneratedWholeContractClauseReview.model_validate(
                    raw_by_id[clause_id]
                )
            except ValidationError as error:
                for issue in self._validation_issues(error):
                    issue.location = ["clause_reviews", index, *issue.location]
                    validation_errors.append(issue)
                clause_reviews.append(
                    WholeContractClauseReviewResult(
                        clause=clause_result.clause,
                        retrieval_status=clause_result.status,
                        status=ClauseReviewStatus.REVIEW_FAILED,
                        error_code="invalid_model_output",
                    )
                )
                continue

            provenance_error = self._validate_clause_provenance(
                generated,
                valid_clause_ids,
                set(sources_by_id),
            )
            if provenance_error is not None:
                validation_errors.append(
                    ReviewValidationIssue(
                        error_type=provenance_error,
                        location=["clause_reviews", index, "provenance"],
                        message=self._provenance_error_message(provenance_error),
                    )
                )
                clause_reviews.append(
                    WholeContractClauseReviewResult(
                        clause=clause_result.clause,
                        retrieval_status=clause_result.status,
                        status=ClauseReviewStatus.REVIEW_FAILED,
                        error_code=provenance_error,
                    )
                )
                continue

            clause_reviews.append(
                self._hydrate_clause_review(
                    generated,
                    clause_result,
                    sources_by_id,
                )
            )

        failed_count = sum(
            review.status == ClauseReviewStatus.REVIEW_FAILED
            for review in clause_reviews
        )
        run_status = (
            ReviewRunStatus.FAILED
            if failed_count == len(clause_reviews)
            else ReviewRunStatus.PARTIAL
        )
        return WholeContractReviewResult(
            document_id=preparation.document_id,
            filename=preparation.filename,
            status=run_status,
            clause_reviews=clause_reviews,
            summary=self._summary(clause_reviews),
            unique_legal_sources=unique_sources,
            evidence_mappings=evidence_mappings,
            llm_debug=WholeContractLLMDebug(
                accepted=False,
                error_code=(
                    "invalid_model_output"
                    if run_status == ReviewRunStatus.FAILED
                    else "partial_invalid_model_output"
                ),
                response=response,
                validation_errors=validation_errors,
            ),
        )

    @staticmethod
    def _validate_clause_provenance(
        generated: GeneratedWholeContractClauseReview,
        valid_clause_ids: set[str],
        valid_source_ids: set[str],
    ) -> str | None:
        assessment_ids = [
            assessment.assessment_id for assessment in generated.assessments
        ]
        if len(assessment_ids) != len(set(assessment_ids)):
            return "duplicate_assessment_id"
        for assessment in generated.assessments:
            if any(
                citation.source_id not in valid_source_ids
                for citation in assessment.citations
            ):
                return "unknown_citation"
            if any(
                clause_id not in valid_clause_ids
                for clause_id in assessment.supporting_contract_clause_ids
            ):
                return "unknown_supporting_contract_clause"
            if (
                generated.clause_id
                not in assessment.supporting_contract_clause_ids
            ):
                return "missing_reviewed_clause_support"
        return None

    @staticmethod
    def _log_model_usage(response: LLMResponse) -> None:
        raw_response = response.raw_response or {}
        raw_usage = raw_response.get("usage")
        metadata = raw_response.get("openrouter_metadata")
        logger.info(
            "whole_contract_llm=%s",
            json.dumps(
                {
                    "request_id": response.request_id,
                    "model": response.model,
                    "provider": response.provider,
                    "finish_reason": response.finish_reason,
                    "usage": (
                        raw_usage
                        if isinstance(raw_usage, dict)
                        else response.usage.model_dump(mode="json")
                    ),
                    "provider_metadata": (
                        metadata if isinstance(metadata, dict) else None
                    ),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )


from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.llm.models import LLMUsage
from app.reviews.models import ContractClause, RetrievalStatus


class ClauseReviewStatus(StrEnum):
    NO_ISSUE_FOUND = "no_issue_found"
    POTENTIAL_VIOLATION = "potential_violation"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"
    NOT_APPLICABLE = "not_applicable"
    REVIEW_FAILED = "review_failed"


class LegalAssessmentOutcome(StrEnum):
    COMPLIANT = "compliant"
    POTENTIAL_VIOLATION = "potential_violation"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"


class AssessmentConfidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GeneratedLegalCitation(BaseModel):
    """Citation requested from the model before backend enrichment."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)


class GeneratedLegalAssessment(BaseModel):
    """A model-generated legal assessment requiring provenance validation."""

    model_config = ConfigDict(extra="forbid")

    assessment_id: str = Field(min_length=1)
    outcome: LegalAssessmentOutcome
    issue: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    citations: list[GeneratedLegalCitation]
    evidence_gap: str | None
    suggested_revision: str | None
    confidence: AssessmentConfidence

    @model_validator(mode="after")
    def validate_outcome_payload(self) -> "GeneratedLegalAssessment":
        if self.outcome == LegalAssessmentOutcome.POTENTIAL_VIOLATION:
            if not self.citations:
                raise ValueError(
                    "potential_violation requires at least one citation"
                )
            if self.evidence_gap is not None:
                raise ValueError(
                    "potential_violation must not include evidence_gap"
                )
            if (
                self.suggested_revision is None
                or not self.suggested_revision.strip()
            ):
                raise ValueError(
                    "potential_violation requires suggested_revision"
                )
            return self

        if self.outcome == LegalAssessmentOutcome.COMPLIANT:
            if not self.citations:
                raise ValueError("compliant requires at least one citation")
            if self.evidence_gap is not None:
                raise ValueError("compliant must not include evidence_gap")
            if self.suggested_revision is not None:
                raise ValueError(
                    "compliant must not include suggested_revision"
                )
            return self

        if self.evidence_gap is None or not self.evidence_gap.strip():
            raise ValueError("needs_more_evidence requires evidence_gap")
        if self.suggested_revision is not None:
            raise ValueError(
                "needs_more_evidence must not include suggested_revision"
            )
        return self


class GeneratedClauseReview(BaseModel):
    """Strict structured output requested from the language model."""

    model_config = ConfigDict(extra="forbid")

    clause_id: str = Field(min_length=1)
    applicable: bool
    applicability_reason: str = Field(min_length=1)
    assessments: list[GeneratedLegalAssessment]

    @model_validator(mode="after")
    def validate_applicability(self) -> "GeneratedClauseReview":
        if self.applicable and not self.assessments:
            raise ValueError(
                "applicable clause requires at least one assessment"
            )
        if not self.applicable and self.assessments:
            raise ValueError(
                "not-applicable clause must not include assessments"
            )
        return self


class LegalCitation(BaseModel):
    """A citation hydrated from a source that retrieval actually returned."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    law_name: str
    section: str
    text: str
    source_file: str | None = None


class LegalAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment_id: str
    outcome: LegalAssessmentOutcome
    issue: str
    reasoning: str
    citations: list[LegalCitation] = Field(default_factory=list)
    evidence_gap: str | None = None
    suggested_revision: str | None = None
    confidence: AssessmentConfidence


class ClauseReviewResult(BaseModel):
    """Trusted application result for one clause."""

    model_config = ConfigDict(extra="forbid")

    clause: ContractClause
    retrieval_status: RetrievalStatus
    status: ClauseReviewStatus
    applicability_reason: str | None = None
    assessments: list[LegalAssessment] = Field(default_factory=list)
    evidence_gap: str | None = None
    error_code: str | None = None
    model: str | None = None
    provider: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)


class ReviewValidationIssue(BaseModel):
    """Serializable validation detail for local developer diagnostics."""

    model_config = ConfigDict(extra="forbid")

    error_type: str
    location: list[str | int] = Field(default_factory=list)
    message: str


class ClauseReviewDiagnostic(BaseModel):
    """Raw LLM response trace written only to the local debug artifact."""

    model_config = ConfigDict(extra="forbid")

    clause_id: str
    accepted: bool
    error_code: str | None = None
    exception_type: str | None = None
    raw_model_output: str | None = None
    validation_errors: list[ReviewValidationIssue] = Field(default_factory=list)
    finish_reason: str | None = None
    request_id: str | None = None
    model: str | None = None
    provider: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)


class ReviewRunStatus(StrEnum):
    READY = "ready"
    PARTIAL = "partial"
    FAILED = "failed"


class ReviewSummary(BaseModel):
    clause_count: int = Field(ge=0)
    assessment_count: int = Field(ge=0)
    compliant_assessment_count: int = Field(ge=0)
    potential_violation_assessment_count: int = Field(ge=0)
    needs_more_evidence_assessment_count: int = Field(ge=0)
    no_issue_found_count: int = Field(ge=0)
    potential_violation_count: int = Field(ge=0)
    needs_more_evidence_count: int = Field(ge=0)
    not_applicable_count: int = Field(ge=0)
    review_failed_count: int = Field(ge=0)


class ContractReviewResult(BaseModel):
    """Clause-level review output; this is not whole-contract certification."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    filename: str
    status: ReviewRunStatus
    clause_reviews: list[ClauseReviewResult] = Field(default_factory=list)
    summary: ReviewSummary

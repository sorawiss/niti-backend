from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.llm.models import LLMUsage
from app.reviews.models import ContractClause, RetrievalStatus


class GeneratedClauseReviewStatus(StrEnum):
    NO_ISSUE_FOUND = "no_issue_found"
    POTENTIAL_VIOLATION = "potential_violation"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"
    NOT_APPLICABLE = "not_applicable"


class ClauseReviewStatus(StrEnum):
    NO_ISSUE_FOUND = "no_issue_found"
    POTENTIAL_VIOLATION = "potential_violation"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"
    NOT_APPLICABLE = "not_applicable"
    REVIEW_FAILED = "review_failed"


class FindingConfidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GeneratedLegalCitation(BaseModel):
    """Citation requested from the model before backend enrichment."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)


class GeneratedLegalFinding(BaseModel):
    """A model-generated finding that still requires provenance validation."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(min_length=1)
    contract_excerpt: str = Field(min_length=1)
    issue: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    citations: list[GeneratedLegalCitation] = Field(min_length=1)
    suggested_revision: str = Field(min_length=1)
    confidence: FindingConfidence


class GeneratedClauseReview(BaseModel):
    """Strict structured output requested from the language model."""

    model_config = ConfigDict(extra="forbid")

    clause_id: str = Field(min_length=1)
    status: GeneratedClauseReviewStatus
    findings: list[GeneratedLegalFinding]
    evidence_gap: str | None

    @model_validator(mode="after")
    def validate_status_payload(self) -> "GeneratedClauseReview":
        if self.status == GeneratedClauseReviewStatus.POTENTIAL_VIOLATION:
            if not self.findings:
                raise ValueError("potential_violation requires at least one finding")
            if self.evidence_gap is not None:
                raise ValueError("potential_violation must not include evidence_gap")
            return self

        if self.findings:
            raise ValueError(f"{self.status.value} must not include findings")

        if self.status == GeneratedClauseReviewStatus.NEEDS_MORE_EVIDENCE:
            if self.evidence_gap is None or not self.evidence_gap.strip():
                raise ValueError("needs_more_evidence requires evidence_gap")
        elif self.evidence_gap is not None:
            raise ValueError(f"{self.status.value} must not include evidence_gap")
        return self


class LegalCitation(BaseModel):
    """A citation hydrated from a source that retrieval actually returned."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    law_name: str
    section: str
    text: str
    source_file: str | None = None


class LegalFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str
    contract_excerpt: str
    issue: str
    reasoning: str
    citations: list[LegalCitation] = Field(min_length=1)
    suggested_revision: str
    confidence: FindingConfidence


class ClauseReviewResult(BaseModel):
    """Trusted application result for one clause."""

    model_config = ConfigDict(extra="forbid")

    clause: ContractClause
    retrieval_status: RetrievalStatus
    status: ClauseReviewStatus
    findings: list[LegalFinding] = Field(default_factory=list)
    evidence_gap: str | None = None
    error_code: str | None = None
    model: str | None = None
    provider: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)


class ReviewRunStatus(StrEnum):
    READY = "ready"
    PARTIAL = "partial"
    FAILED = "failed"


class ReviewSummary(BaseModel):
    clause_count: int = Field(ge=0)
    finding_count: int = Field(ge=0)
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

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.llm.models import LLMResponse
from app.reviews.models import (
    ContractClause,
    QueryRetrievalResult,
    RetrievedLegalSource,
    RetrievalStatus,
)
from app.reviews.review_models import (
    AssessmentConfidence,
    ClauseReviewStatus,
    GeneratedClauseReview,
    GeneratedLegalCitation,
    LegalAssessmentOutcome,
    LegalCitation,
    ReviewRunStatus,
    ReviewSummary,
    ReviewValidationIssue,
)


class GeneratedWholeContractAssessment(BaseModel):
    """Generated assessment with explicit contract-clause provenance."""

    model_config = ConfigDict(extra="forbid")

    assessment_id: str = Field(min_length=1)
    outcome: LegalAssessmentOutcome
    issue: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    citations: list[GeneratedLegalCitation]
    confidence: AssessmentConfidence
    supporting_contract_clause_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_required_citations(self) -> "GeneratedWholeContractAssessment":
        if self.outcome != LegalAssessmentOutcome.NEEDS_MORE_EVIDENCE:
            if not self.citations:
                raise ValueError(f"{self.outcome.value} requires a citation")
        return self


class GeneratedWholeContractClauseReview(GeneratedClauseReview):
    assessments: list[GeneratedWholeContractAssessment]


class ContractFindingType(StrEnum):
    MISSING_TERM = "missing_term"
    CROSS_CLAUSE_CONFLICT = "cross_clause_conflict"
    CONTRACT_WIDE_RISK = "contract_wide_risk"
    DRAFTING_ISSUE = "drafting_issue"


class GeneratedContractFinding(BaseModel):
    """A material finding that cannot honestly belong to one clause review."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(min_length=1)
    finding_type: ContractFindingType
    outcome: LegalAssessmentOutcome
    issue: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    citations: list[GeneratedLegalCitation]
    confidence: AssessmentConfidence
    supporting_contract_clause_ids: list[str]

    @model_validator(mode="after")
    def validate_finding(self) -> "GeneratedContractFinding":
        if (
            self.outcome != LegalAssessmentOutcome.NEEDS_MORE_EVIDENCE
            and not self.citations
        ):
            raise ValueError(f"{self.outcome.value} requires a citation")
        if (
            self.finding_type == ContractFindingType.MISSING_TERM
            and self.supporting_contract_clause_ids
        ):
            raise ValueError("missing_term must not cite a contract clause")
        if (
            self.finding_type != ContractFindingType.MISSING_TERM
            and not self.supporting_contract_clause_ids
        ):
            raise ValueError(
                f"{self.finding_type.value} requires a supporting contract clause"
            )
        return self


class GeneratedWholeContractReview(BaseModel):
    """The single structured response requested for the entire contract."""

    model_config = ConfigDict(extra="forbid")

    clause_reviews: list[GeneratedWholeContractClauseReview]
    contract_findings: list[GeneratedContractFinding]
    overall_summary: str = Field(min_length=1)


class ContractFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str
    finding_type: ContractFindingType
    outcome: LegalAssessmentOutcome
    issue: str
    reasoning: str
    citations: list[LegalCitation] = Field(default_factory=list)
    confidence: AssessmentConfidence
    supporting_contract_clause_ids: list[str] = Field(default_factory=list)


class WholeContractLegalAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment_id: str
    outcome: LegalAssessmentOutcome
    issue: str
    reasoning: str
    citations: list[LegalCitation] = Field(default_factory=list)
    confidence: AssessmentConfidence
    supporting_contract_clause_ids: list[str] = Field(min_length=1)


class WholeContractClauseReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clause: ContractClause
    retrieval_status: RetrievalStatus
    status: ClauseReviewStatus
    query_results: list[QueryRetrievalResult] = Field(default_factory=list)
    applicability_reason: str | None = None
    assessments: list[WholeContractLegalAssessment] = Field(default_factory=list)
    error_code: str | None = None


class ClauseQuerySourceMapping(BaseModel):
    """Preserves the exact clause -> query -> retrieved-source relationship."""

    model_config = ConfigDict(extra="forbid")

    clause_id: str
    query_id: str
    query_text: str
    retrieval_status: RetrievalStatus
    source_ids: list[str] = Field(default_factory=list)
    error_code: str | None = None


class WholeContractLLMDebug(BaseModel):
    """Complete normalized gateway response plus backend validation details."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    error_code: str | None = None
    exception_type: str | None = None
    response: LLMResponse | None = None
    validation_errors: list[ReviewValidationIssue] = Field(default_factory=list)


class WholeContractReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    filename: str
    status: ReviewRunStatus
    clause_reviews: list[WholeContractClauseReviewResult] = Field(
        default_factory=list
    )
    contract_findings: list[ContractFinding] = Field(default_factory=list)
    overall_summary: str | None = None
    summary: ReviewSummary
    unique_legal_sources: list[RetrievedLegalSource] = Field(default_factory=list)
    evidence_mappings: list[ClauseQuerySourceMapping] = Field(default_factory=list)
    llm_debug: WholeContractLLMDebug


class WholeContractReviewPublicResult(BaseModel):
    """Validated API payload; raw model output remains a local debug artifact."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    filename: str
    status: ReviewRunStatus
    clause_reviews: list[WholeContractClauseReviewResult] = Field(
        default_factory=list
    )
    contract_findings: list[ContractFinding] = Field(default_factory=list)
    overall_summary: str | None = None
    summary: ReviewSummary
    unique_legal_sources: list[RetrievedLegalSource] = Field(default_factory=list)
    evidence_mappings: list[ClauseQuerySourceMapping] = Field(default_factory=list)

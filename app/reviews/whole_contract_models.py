from pydantic import BaseModel, ConfigDict, Field

from app.llm.models import LLMResponse
from app.reviews.models import (
    ContractClause,
    RetrievedLegalSource,
    RetrievalStatus,
)
from app.reviews.review_models import (
    ClauseReviewStatus,
    GeneratedClauseReview,
    GeneratedLegalAssessment,
    LegalAssessment,
    ReviewRunStatus,
    ReviewSummary,
    ReviewValidationIssue,
)


class GeneratedWholeContractAssessment(GeneratedLegalAssessment):
    """Generated assessment with explicit contract-clause provenance."""

    supporting_contract_clause_ids: list[str] = Field(min_length=1)


class GeneratedWholeContractClauseReview(GeneratedClauseReview):
    assessments: list[GeneratedWholeContractAssessment]


class GeneratedWholeContractReview(BaseModel):
    """The single structured response requested for the entire contract."""

    model_config = ConfigDict(extra="forbid")

    clause_reviews: list[GeneratedWholeContractClauseReview]


class WholeContractLegalAssessment(LegalAssessment):
    supporting_contract_clause_ids: list[str] = Field(min_length=1)


class WholeContractClauseReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clause: ContractClause
    retrieval_status: RetrievalStatus
    status: ClauseReviewStatus
    applicability_reason: str | None = None
    assessments: list[WholeContractLegalAssessment] = Field(default_factory=list)
    evidence_gap: str | None = None
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
    summary: ReviewSummary
    unique_legal_sources: list[RetrievedLegalSource] = Field(default_factory=list)
    evidence_mappings: list[ClauseQuerySourceMapping] = Field(default_factory=list)

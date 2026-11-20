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


class CompletenessStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"
    UNCERTAIN = "uncertain"


class ContractCompletenessCategory(StrEnum):
    PARTIES_INFORMATION = "parties_information"
    POSITION_AND_DUTIES = "position_and_duties"
    EMPLOYMENT_DURATION = "employment_duration"
    COMPENSATION_AND_BENEFITS = "compensation_and_benefits"
    WORKING_HOURS_AND_HOLIDAYS = "working_hours_and_holidays"
    CONFIDENTIALITY = "confidentiality"
    TERMINATION = "termination"
    DISPUTE_RESOLUTION = "dispute_resolution"


COMPLETENESS_CATEGORY_LABELS: dict[ContractCompletenessCategory, str] = {
    ContractCompletenessCategory.PARTIES_INFORMATION: "ข้อมูลคู่สัญญา",
    ContractCompletenessCategory.POSITION_AND_DUTIES: "ตำแหน่งและหน้าที่",
    ContractCompletenessCategory.EMPLOYMENT_DURATION: "ระยะเวลาการจ้างงาน",
    ContractCompletenessCategory.COMPENSATION_AND_BENEFITS: (
        "ค่าตอบแทนและสวัสดิการ"
    ),
    ContractCompletenessCategory.WORKING_HOURS_AND_HOLIDAYS: (
        "เวลาทำงานและวันหยุด"
    ),
    ContractCompletenessCategory.CONFIDENTIALITY: "การรักษาความลับ",
    ContractCompletenessCategory.TERMINATION: "การเลิกจ้าง",
    ContractCompletenessCategory.DISPUTE_RESOLUTION: "การระงับข้อพิพาท",
}

class ContractCompletenessItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: ContractCompletenessCategory
    label: str = Field(min_length=1)
    status: CompletenessStatus
    summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_fixed_label_and_summary(self) -> "ContractCompletenessItem":
        expected_label = COMPLETENESS_CATEGORY_LABELS[self.category]
        if self.label != expected_label:
            raise ValueError(
                f"label for {self.category.value} must be {expected_label!r}"
            )
        if not self.summary.strip():
            raise ValueError("summary must not be blank")
        return self


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


class GeneratedWholeContractReview(BaseModel):
    """The single structured response requested for the entire contract."""

    model_config = ConfigDict(extra="forbid")

    clause_reviews: list[GeneratedWholeContractClauseReview]
    completeness: list[ContractCompletenessItem] = Field(
        min_length=len(ContractCompletenessCategory),
        max_length=len(ContractCompletenessCategory),
    )

    @model_validator(mode="after")
    def validate_completeness_categories(self) -> "GeneratedWholeContractReview":
        expected = list(ContractCompletenessCategory)
        returned = [item.category for item in self.completeness]
        if returned != expected:
            raise ValueError(
                "completeness categories must appear exactly once in the "
                "required order"
            )
        return self


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
    summary: ReviewSummary
    completeness: list[ContractCompletenessItem] = Field(default_factory=list)
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
    completeness: list[ContractCompletenessItem] = Field(default_factory=list)
    unique_legal_sources: list[RetrievedLegalSource] = Field(default_factory=list)
    evidence_mappings: list[ClauseQuerySourceMapping] = Field(default_factory=list)

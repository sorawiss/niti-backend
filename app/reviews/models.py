from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class BoundaryReason(StrEnum):
    DOCUMENT_START = "document_start"
    HEADING = "heading"
    NUMBERED_CLAUSE = "numbered_clause"
    HEADING_AND_NUMBERED = "heading_and_numbered"
    UNSTRUCTURED_BLOCK = "unstructured_block"


class ContractClause(BaseModel):
    """One reviewable contract unit mapped back to parser blocks."""

    model_config = ConfigDict(extra="forbid")

    clause_id: str
    order: int = Field(ge=0)
    number: str | None = None
    title: str | None = None
    text: str = Field(min_length=1)
    block_indices: list[int] = Field(min_length=1)
    parent_clause_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    boundary_reason: BoundaryReason


class QueryStrategy(StrEnum):
    CLAUSE_TEXT = "clause_text"


class ClauseQuery(BaseModel):
    """A deterministic retrieval query for one contract clause."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    clause_id: str
    text: str = Field(min_length=1)
    block_indices: list[int] = Field(min_length=1)
    strategy: QueryStrategy = QueryStrategy.CLAUSE_TEXT


class RetrievalStatus(StrEnum):
    READY = "ready"
    PARTIAL = "partial"
    NO_RESULTS = "no_results"
    FAILED = "failed"


class SourceMatch(BaseModel):
    query_id: str
    rank: int = Field(ge=1)
    score: float | None = None


class RetrievedLegalSource(BaseModel):
    """A stable legal source with all query matches that discovered it."""

    source_id: str
    law_name: str
    section: str
    text: str
    source_file: str | None = None
    best_score: float | None = None
    best_rank: int = Field(ge=1)
    matched_query_ids: list[str] = Field(default_factory=list)
    matches: list[SourceMatch] = Field(default_factory=list)


class QueryRetrievalResult(BaseModel):
    query_id: str
    status: RetrievalStatus
    source_ids: list[str] = Field(default_factory=list)
    invalid_result_count: int = Field(default=0, ge=0)
    error_code: str | None = None


class ClauseRetrievalResult(BaseModel):
    clause: ContractClause
    queries: list[ClauseQuery] = Field(min_length=1)
    query_results: list[QueryRetrievalResult] = Field(min_length=1)
    sources: list[RetrievedLegalSource] = Field(default_factory=list)
    status: RetrievalStatus


class PreparationStatus(StrEnum):
    READY = "ready"
    PARTIAL = "partial"
    FAILED = "failed"


class PreparationSummary(BaseModel):
    block_count: int = Field(ge=0)
    clause_count: int = Field(ge=0)
    query_count: int = Field(ge=0)
    retrieved_source_count: int = Field(ge=0)
    ready_clause_count: int = Field(ge=0)
    partial_clause_count: int = Field(ge=0)
    no_results_clause_count: int = Field(ge=0)
    failed_clause_count: int = Field(ge=0)


class ClauseRetrievalPreparation(BaseModel):
    """Traceable retrieval evidence; this is not a legal judgment."""

    document_id: str
    filename: str
    status: PreparationStatus
    clause_results: list[ClauseRetrievalResult] = Field(default_factory=list)
    summary: PreparationSummary


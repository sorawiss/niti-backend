from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.llm.models import LLMUsage


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
BlockIndex = Annotated[int, Field(strict=True, ge=0)]


class ReviewModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievalQuery(ReviewModel):
    query_id: NonEmptyText
    query: NonEmptyText


class ReviewTopic(ReviewModel):
    topic_id: NonEmptyText
    description: NonEmptyText
    block_indices: list[BlockIndex]
    queries: list[RetrievalQuery] = Field(min_length=1)


class ContextDisposition(ReviewModel):
    """A block considered as context rather than a separate review topic."""

    block_index: BlockIndex
    reason: NonEmptyText


class ReviewPlan(ReviewModel):
    topics: list[ReviewTopic] = Field(min_length=1)
    context_blocks: list[ContextDisposition]


class SourceMatch(ReviewModel):
    query_id: str
    score: float | None = Field(default=None, allow_inf_nan=False)


class ReviewSource(ReviewModel):
    source_id: str
    law_name: str
    section: str
    text: str
    source_files: list[str] = Field(default_factory=list)
    matches: list[SourceMatch] = Field(default_factory=list)


class QueryRetrieval(ReviewModel):
    query_id: str
    query: str
    status: Literal["retrieved", "no_results", "failed", "partial"]
    source_ids: list[str] = Field(default_factory=list)
    invalid_result_count: int = 0
    error_code: str | None = None


class RetrievalBundle(ReviewModel):
    sources: list[ReviewSource]
    queries: list[QueryRetrieval]


class ReviewPreparation(ReviewModel):
    """An inspectable retrieval plan, not a legal assessment."""

    document_id: str
    filename: str
    status: Literal["ready", "partial"]
    plan: ReviewPlan
    retrieval: RetrievalBundle
    model: str
    provider: str | None
    usage: LLMUsage

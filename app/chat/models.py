from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.llm.models import LLMUsage
from app.rag.rewrite import QueryRewriteResult


class LegalAnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class LegalSource(BaseModel):
    source_id: str
    score: float | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None
    retrieval_rank: int | None = None
    rerank_rank: int | None = None
    law_name: str
    section: str
    text: str
    source_file: str | None = None


class GeneratedLegalAnswer(BaseModel):
    """Strict structure requested from the language model."""

    model_config = ConfigDict(extra="forbid")

    status: LegalAnswerStatus
    answer: str = Field(min_length=1)
    citation_ids: list[str]
    requires_human_review: bool


MAX_SUPPLEMENTAL_SEARCHES = 3


class ChatSearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[
        Annotated[
            str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
        ]
    ] = Field(max_length=MAX_SUPPLEMENTAL_SEARCHES)


class ChatWorkflowResult(BaseModel):
    status: LegalAnswerStatus
    answer: str
    citations: list[LegalSource]
    requires_human_review: bool
    model: str | None = None
    provider: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)
    rewrite: QueryRewriteResult

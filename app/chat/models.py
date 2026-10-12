from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.llm.models import LLMUsage


class LegalAnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class LegalSource(BaseModel):
    source_id: str
    score: float | None = None
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


class ChatWorkflowResult(BaseModel):
    status: LegalAnswerStatus
    answer: str
    citations: list[LegalSource]
    requires_human_review: bool
    model: str | None = None
    provider: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: ChatRole
    content: str = Field(min_length=1)


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_output_tokens: int | None = Field(default=None, ge=1)
    response_format: dict[str, Any] | None = None
    user_id: str | None = None


class LLMUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None


class LLMResponse(BaseModel):
    content: str
    model: str
    provider: str | None = None
    finish_reason: str | None = None
    request_id: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)

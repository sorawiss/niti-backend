from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.chat.models import LegalAnswerStatus
from app.chat.service import ChatGenerationError, ChatService
from app.llm.errors import LLMGatewayError


router = APIRouter(tags=["Chat"])


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=10_000,
        description="One Thai legal question. No conversation history is stored.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of law chunks supplied to the language model.",
    )
    law_name: str | None = Field(
        default=None,
        description="Optional exact law name metadata filter.",
    )


class ChatCitation(BaseModel):
    source_id: str
    score: float | None
    law_name: str
    section: str
    text: str
    source_file: str | None


class ChatUsage(BaseModel):
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost: float | None


class ChatResponse(BaseModel):
    status: LegalAnswerStatus
    answer: str
    citations: list[ChatCitation]
    requires_human_review: bool
    model: str | None
    provider: str | None
    usage: ChatUsage


def get_chat_service(request: Request) -> ChatService:
    service = getattr(request.app.state, "chat_service", None)
    if not isinstance(service, ChatService):
        raise RuntimeError("Chat service is not initialized")
    return service


ChatServiceDependency = Annotated[ChatService, Depends(get_chat_service)]


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, service: ChatServiceDependency) -> ChatResponse:
    try:
        result = await service.answer(
            request.message,
            top_k=request.top_k,
            law_name=request.law_name,
        )
        return ChatResponse(
            status=result.status,
            answer=result.answer,
            citations=[
                ChatCitation.model_validate(citation.model_dump())
                for citation in result.citations
            ],
            requires_human_review=result.requires_human_review,
            model=result.model,
            provider=result.provider,
            usage=ChatUsage.model_validate(result.usage.model_dump()),
        )
    except LLMGatewayError as error:
        raise HTTPException(
            status_code=503 if error.retryable else 502,
            detail={
                "code": error.code,
                "message": "บริการ AI ไม่พร้อมใช้งาน กรุณาลองใหม่ภายหลัง",
                "retryable": error.retryable,
            },
        ) from error
    except ChatGenerationError as error:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "invalid_ai_response",
                "message": "ระบบไม่สามารถตรวจสอบรูปแบบคำตอบจาก AI ได้",
                "retryable": True,
            },
        ) from error
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "chat_workflow_failed",
                "message": "ไม่สามารถประมวลผลคำถามได้",
                "retryable": False,
            },
        ) from error

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from app.chat.models import (
    ChatWorkflowResult,
    GeneratedLegalAnswer,
    LegalAnswerStatus,
    LegalSource,
)
from app.chat.prompts import build_rag_messages
from app.llm.base import LLMGateway
from app.llm.models import LLMRequest
from app.rag.query import retrieve_laws


Retriever = Callable[..., list[dict[str, Any]]]


class ChatGenerationError(Exception):
    """Raised when an otherwise successful LLM response cannot be trusted."""


class ChatService:
    def __init__(
        self,
        llm_gateway: LLMGateway,
        retriever: Retriever = retrieve_laws,
    ) -> None:
        self._llm_gateway = llm_gateway
        self._retriever = retriever

    async def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        law_name: str | None = None,
    ) -> ChatWorkflowResult:
        raw_results = await asyncio.to_thread(
            self._retriever,
            query=question,
            top_k=top_k,
            law_name=law_name,
        )
        sources = self._to_legal_sources(raw_results)

        if not sources:
            return ChatWorkflowResult(
                status=LegalAnswerStatus.INSUFFICIENT_CONTEXT,
                answer="ไม่พบข้อมูลกฎหมายที่เกี่ยวข้องเพียงพอสำหรับตอบคำถามนี้",
                citations=[],
                requires_human_review=True,
            )

        llm_response = await self._llm_gateway.generate(
            LLMRequest(
                messages=build_rag_messages(question, sources),
                response_format=self._response_format(),
            )
        )

        try:
            generated = GeneratedLegalAnswer.model_validate_json(llm_response.content)
        except ValidationError as error:
            raise ChatGenerationError(
                "The language model did not return a valid legal answer"
            ) from error

        sources_by_id = {source.source_id: source for source in sources}
        valid_citation_ids = list(
            dict.fromkeys(
                source_id
                for source_id in generated.citation_ids
                if source_id in sources_by_id
            )
        )
        citations = [sources_by_id[source_id] for source_id in valid_citation_ids]

        if generated.status == LegalAnswerStatus.ANSWERED and not citations:
            return ChatWorkflowResult(
                status=LegalAnswerStatus.INSUFFICIENT_CONTEXT,
                answer=(
                    "ระบบไม่สามารถยืนยันแหล่งอ้างอิงของคำตอบได้ "
                    "จึงยังไม่ควรใช้คำตอบนี้เป็นข้อมูลทางกฎหมาย"
                ),
                citations=[],
                requires_human_review=True,
                model=llm_response.model,
                provider=llm_response.provider,
                usage=llm_response.usage,
            )

        return ChatWorkflowResult(
            status=generated.status,
            answer=generated.answer,
            citations=citations,
            requires_human_review=generated.requires_human_review,
            model=llm_response.model,
            provider=llm_response.provider,
            usage=llm_response.usage,
        )

    @staticmethod
    def _to_legal_sources(raw_results: list[dict[str, Any]]) -> list[LegalSource]:
        sources: list[LegalSource] = []
        for index, result in enumerate(raw_results, start=1):
            metadata = result.get("metadata") or {}
            text = result.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            sources.append(
                LegalSource(
                    source_id=f"LAW_{index}",
                    score=result.get("score"),
                    law_name=str(metadata.get("law_name") or "ไม่ระบุชื่อกฎหมาย"),
                    section=str(metadata.get("section") or "ไม่ระบุมาตรา"),
                    text=text,
                    source_file=metadata.get("source_file"),
                )
            )
        return sources

    @staticmethod
    def _response_format() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "legal_chat_answer",
                "strict": True,
                "schema": GeneratedLegalAnswer.model_json_schema(),
            },
        }

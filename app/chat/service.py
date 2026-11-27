import asyncio
import logging
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.chat.models import (
    ChatSearchPlan,
    ChatWorkflowResult,
    GeneratedLegalAnswer,
    LegalAnswerStatus,
    LegalSource,
)
from app.chat.prompts import build_rag_messages, build_search_plan_messages
from app.llm.base import LLMGateway
from app.llm.config import QueryRewriteWorkflow
from app.llm.errors import LLMGatewayError
from app.llm.models import LLMRequest
from app.rag.query import retrieve_laws
from app.rag.rewrite import QueryRewriteResult, QueryRewriteService


Retriever = Callable[..., list[dict[str, Any]]]
logger = logging.getLogger(__name__)


class ChatState(TypedDict, total=False):
    question: str
    top_k: int
    initial_law_name: str | None
    rewrite: QueryRewriteResult
    sources: list[LegalSource]
    plan: ChatSearchPlan
    result: ChatWorkflowResult


class ChatGenerationError(Exception):
    """Raised when an otherwise successful LLM response cannot be trusted."""


class ChatService:
    def __init__(
        self,
        llm_gateway: LLMGateway,
        retriever: Retriever = retrieve_laws,
        query_rewriter: QueryRewriteService | None = None,
    ) -> None:
        self._llm_gateway = llm_gateway
        self._retriever = retriever
        self._query_rewriter = query_rewriter
        graph = StateGraph(ChatState)
        graph.add_node("plan_searches", self._plan_searches)
        graph.add_node("retrieve_evidence", self._retrieve_evidence)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_edge(START, "plan_searches")
        graph.add_edge("plan_searches", "retrieve_evidence")
        graph.add_edge("retrieve_evidence", "generate_answer")
        graph.add_edge("generate_answer", END)
        self._graph = graph.compile()

    async def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        law_name: str | None = None,
    ) -> ChatWorkflowResult:
        rewrite = (
            await self._query_rewriter.rewrite(
                question,
                workflow=QueryRewriteWorkflow.CHAT,
            )
            if self._query_rewriter is not None
            else QueryRewriteResult.disabled(question)
        )
        raw_results = await asyncio.to_thread(
            self._retriever,
            query=rewrite.effective_query,
            top_k=top_k,
            law_name=law_name,
        )
        state = await self._graph.ainvoke(
            {
                "question": question,
                "top_k": top_k,
                "initial_law_name": law_name,
                "rewrite": rewrite,
                "sources": self._to_legal_sources(raw_results),
            }
        )
        return state["result"]

    async def _plan_searches(self, state: ChatState) -> ChatState:
        try:
            response = await self._llm_gateway.generate(
                LLMRequest(
                    messages=build_search_plan_messages(
                        state["question"],
                        state["rewrite"].effective_query,
                        state["sources"],
                        state["initial_law_name"],
                    ),
                    max_output_tokens=4096,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "legal_chat_search_plan",
                            "strict": True,
                            "schema": ChatSearchPlan.model_json_schema(),
                        },
                    },
                )
            )
            plan = ChatSearchPlan.model_validate_json(response.content)
        except (LLMGatewayError, ValidationError) as error:
            error_code = (
                error.code if isinstance(error, LLMGatewayError) else type(error).__name__
            )
            logger.warning("Chat search planning failed (%s)", error_code)
            plan = ChatSearchPlan(queries=[])
        return {"plan": plan}

    async def _retrieve_evidence(self, state: ChatState) -> ChatState:
        sources = state["sources"]
        seen: set[str] = set()
        if not state["initial_law_name"]:
            seen.add(" ".join(state["rewrite"].effective_query.split()).casefold())
        for query in state["plan"].queries:
            key = " ".join(query.split()).casefold()
            if key in seen:
                continue
            seen.add(key)
            try:
                raw_results = await asyncio.to_thread(
                    self._retriever,
                    query=query,
                    top_k=min(state["top_k"], 5),
                    law_name=None,
                )
                sources = self._to_legal_sources(raw_results, existing=sources)
            except Exception as error:
                logger.warning(
                    "Supplemental chat retrieval failed (%s)", type(error).__name__
                )
        return {"sources": sources}

    async def _generate_answer(self, state: ChatState) -> ChatState:
        return {
            "result": await self._answer_from_sources(
                state["question"], state["sources"], state["rewrite"]
            )
        }

    async def _answer_from_sources(
        self,
        question: str,
        sources: list[LegalSource],
        rewrite: QueryRewriteResult,
    ) -> ChatWorkflowResult:
        if not sources:
            return ChatWorkflowResult(
                status=LegalAnswerStatus.INSUFFICIENT_CONTEXT,
                answer="ไม่พบข้อมูลกฎหมายที่เกี่ยวข้องเพียงพอสำหรับตอบคำถามนี้",
                citations=[],
                requires_human_review=True,
                rewrite=rewrite,
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
                rewrite=rewrite,
            )

        return ChatWorkflowResult(
            status=generated.status,
            answer=generated.answer,
            citations=citations,
            requires_human_review=generated.requires_human_review,
            model=llm_response.model,
            provider=llm_response.provider,
            usage=llm_response.usage,
            rewrite=rewrite,
        )

    @staticmethod
    def _to_legal_sources(
        raw_results: list[dict[str, Any]],
        *,
        existing: list[LegalSource] | None = None,
    ) -> list[LegalSource]:
        sources = list(existing or [])
        seen = {
            (source.law_name, source.section, source.text, source.source_file)
            for source in sources
        }
        for result in raw_results:
            metadata = result.get("metadata") or {}
            text = result.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            law_name = str(metadata.get("law_name") or "ไม่ระบุชื่อกฎหมาย")
            section = str(metadata.get("section") or "ไม่ระบุมาตรา")
            source_file = metadata.get("source_file")
            key = (law_name, section, text, source_file)
            if key in seen:
                continue
            sources.append(
                LegalSource(
                    source_id=f"LAW_{len(sources) + 1}",
                    score=result.get("score"),
                    retrieval_score=result.get("retrieval_score"),
                    rerank_score=result.get("rerank_score"),
                    retrieval_rank=result.get("retrieval_rank"),
                    rerank_rank=result.get("rerank_rank"),
                    law_name=law_name,
                    section=section,
                    text=text,
                    source_file=source_file,
                )
            )
            seen.add(key)
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

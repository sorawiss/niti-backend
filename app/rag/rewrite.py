import asyncio
import json
import logging
import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.llm.base import LLMGateway
from app.llm.config import QueryRewriteSettings, QueryRewriteWorkflow
from app.llm.errors import LLMGatewayError
from app.llm.models import (
    ChatMessage,
    ChatRole,
    LLMRequest,
    LLMResponse,
    LLMUsage,
)


logger = logging.getLogger("uvicorn.error")


class QueryRewriteStatus(StrEnum):
    DISABLED = "disabled"
    REWRITTEN = "rewritten"
    FALLBACK_ORIGINAL = "fallback_original"


class QueryRewriteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: QueryRewriteStatus
    original_query: str
    effective_query: str
    model: str | None = None
    provider: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)
    error_code: str | None = None

    @classmethod
    def disabled(cls, query: str) -> "QueryRewriteResult":
        return cls(
            status=QueryRewriteStatus.DISABLED,
            original_query=query,
            effective_query=query,
        )


class _GeneratedRewrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rewritten_query: str = Field(min_length=1)


SYSTEM_PROMPT = """You are a query rewriting assistant for a Thai labor law document retrieval system.
Your task is to transform the given input into a clear, concise, and semantically rich Thai search query suitable for embedding with the BGE-M3 model.
Instructions:
1. Extract the essential legal concepts, rights, obligations, conditions, and relevant employment context from the input.
2. Remove unnecessary details that do not contribute to legal document retrieval, such as company names, personal names, specific organization names, and unrelated contextual information.
3. Retain specific details such as employment conditions, monetary amounts, durations, exceptions, and other facts when they are relevant to identifying applicable labor laws.
4. Rewrite the extracted information using clear and appropriate Thai legal terminology.
5. Express the output as a declarative statement, not as a question, regardless of the input format.
6. Use appropriate legal terminology and synonyms to improve retrieval relevance without changing the intended legal meaning.
7. Do not introduce unsupported facts, legal provisions, or interpretations.
8. Do not answer the input, explain legal concepts, analyze legal compliance, or provide legal advice.
9. Always write the output in Thai, regardless of the input language.
Output only the rewritten Thai statement as plain text, without headings, explanations, or additional commentary.
If the input contains no meaningful information relevant to Thai labor law retrieval, preserve its essential meaning without inventing legal context."""


class QueryRewriteService:
    def __init__(
        self,
        llm_gateway: LLMGateway,
        settings: QueryRewriteSettings,
    ) -> None:
        self._llm_gateway = llm_gateway
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def rewrite(
        self,
        query: str,
        *,
        workflow: QueryRewriteWorkflow,
        user_id: str | None = None,
    ) -> QueryRewriteResult:
        if not self._settings.is_enabled(workflow):
            result = QueryRewriteResult.disabled(query)
            self._log(
                workflow,
                result,
                duration_ms=0,
                response=None,
                llm_called=False,
                error_raw_response=None,
            )
            return result

        started = time.perf_counter()
        response: LLMResponse | None = None
        error_raw_response: Any | None = None
        try:
            async with self._semaphore:
                response = await asyncio.wait_for(
                    self._llm_gateway.generate(
                        LLMRequest(
                            messages=[
                                ChatMessage(
                                    role=ChatRole.SYSTEM,
                                    content=SYSTEM_PROMPT,
                                ),
                                ChatMessage(
                                    role=ChatRole.USER,
                                    content=json.dumps(
                                        {"query": query}, ensure_ascii=False
                                    ),
                                ),
                            ],
                            model=self._settings.model,
                            temperature=0,
                            max_output_tokens=self._settings.max_output_tokens,
                            response_format=self._response_format(),
                            user_id=user_id,
                        )
                    ),
                    timeout=self._settings.timeout_seconds,
                )
            generated = _GeneratedRewrite.model_validate_json(response.content)
            rewritten_query = generated.rewritten_query.strip()
            if (
                not rewritten_query
                or len(rewritten_query) > self._settings.max_query_chars
            ):
                raise ValueError("rewritten query is empty or too long")
            result = QueryRewriteResult(
                status=QueryRewriteStatus.REWRITTEN,
                original_query=query,
                effective_query=rewritten_query,
                model=response.model,
                provider=response.provider,
                usage=response.usage,
            )
        except TimeoutError:
            result = self._fallback(query, "query_rewrite_timeout", response)
        except LLMGatewayError as error:
            error_raw_response = error.raw_response
            result = self._fallback(query, error.code, response)
        except (ValidationError, ValueError, TypeError):
            result = self._fallback(query, "invalid_rewrite", response)
        except Exception:
            result = self._fallback(
                query, "query_rewrite_unexpected_error", response
            )

        self._log(
            workflow,
            result,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            response=response,
            llm_called=True,
            error_raw_response=error_raw_response,
        )
        return result

    @staticmethod
    def _fallback(
        query: str,
        error_code: str,
        response: LLMResponse | None,
    ) -> QueryRewriteResult:
        return QueryRewriteResult(
            status=QueryRewriteStatus.FALLBACK_ORIGINAL,
            original_query=query,
            effective_query=query,
            model=response.model if response else None,
            provider=response.provider if response else None,
            usage=response.usage if response else LLMUsage(),
            error_code=error_code,
        )

    @staticmethod
    def _response_format() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "query_rewrite",
                "strict": True,
                "schema": _GeneratedRewrite.model_json_schema(),
            },
        }

    def _log(
        self,
        workflow: QueryRewriteWorkflow,
        result: QueryRewriteResult,
        *,
        duration_ms: float,
        response: LLMResponse | None,
        llm_called: bool,
        error_raw_response: Any | None,
    ) -> None:
        raw_response = (
            response.raw_response if response is not None else error_raw_response
        )
        raw_response_data = (
            raw_response if isinstance(raw_response, dict) else {}
        )
        raw_usage = raw_response_data.get("usage")
        provider_metadata = raw_response_data.get("openrouter_metadata")
        payload: dict[str, Any] = {
            "event": (
                "query_rewrite_llm_completed"
                if llm_called
                else "query_rewrite_disabled"
            ),
            "workflow": workflow.value,
            "llm_called": llm_called,
            "status": result.status.value,
            "duration_ms": duration_ms,
            "request_id": response.request_id if response else None,
            "model": (
                response.model
                if response
                else self._settings.model if llm_called else None
            ),
            "provider": result.provider,
            "finish_reason": response.finish_reason if response else None,
            "usage": (
                raw_usage
                if isinstance(raw_usage, dict)
                else result.usage.model_dump(mode="json")
            ),
            "provider_metadata": (
                provider_metadata
                if isinstance(provider_metadata, dict)
                else None
            ),
            "error_code": result.error_code,
        }
        if self._settings.log_raw_response and llm_called:
            payload["raw_response"] = raw_response

        logger.info(
            "query_rewrite=%s",
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

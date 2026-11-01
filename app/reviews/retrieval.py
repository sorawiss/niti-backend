import asyncio
import hashlib
import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.reviews.models import (
    QueryRetrieval,
    RetrievalBundle,
    RetrievalQuery,
    ReviewPlan,
    ReviewSource,
    SourceMatch,
)


Retriever = Callable[..., list[dict[str, Any]]]


def _retrieve_laws(**kwargs: Any) -> list[dict[str, Any]]:
    # Keep the embedding model and Qdrant imports out of network-free tests.
    from app.rag.query import retrieve_laws

    return retrieve_laws(**kwargs)


class _Metadata(BaseModel):
    model_config = ConfigDict(extra="ignore")
    law_name: str
    section: str
    source_file: str | None = None

    @field_validator("law_name", "section")
    @classmethod
    def require_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Missing legal provision identity")
        return value


class _RetrievedLaw(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str
    metadata: _Metadata
    score: float | None = Field(default=None, allow_inf_nan=False)

    @field_validator("text")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Empty law text")
        return value


def source_id_for(law_name: str, section: str, text: str) -> str:
    identity = json.dumps([law_name, section, text], ensure_ascii=False).encode("utf-8")
    return "src_" + hashlib.sha256(identity).hexdigest()


class ReviewRetriever:
    def __init__(
        self,
        retriever: Retriever = _retrieve_laws,
        *,
        top_k: int = 5,
        concurrency: int = 4,
        max_queries: int = 32,
    ) -> None:
        if not 1 <= top_k <= 20 or concurrency < 1 or max_queries < 1:
            raise ValueError("Invalid retrieval limits")
        self._retriever = retriever
        self._top_k = top_k
        self._concurrency = concurrency
        self._max_queries = max_queries

    async def retrieve(self, plan: ReviewPlan) -> RetrievalBundle:
        queries = [query for topic in plan.topics for query in topic.queries]
        if len(queries) > self._max_queries:
            raise ValueError("Review query limit exceeded")
        if len({query.query_id for query in queries}) != len(queries):
            raise ValueError("Duplicate query IDs")
        semaphore = asyncio.Semaphore(self._concurrency)

        async def search(query: RetrievalQuery) -> tuple[list[_RetrievedLaw], QueryRetrieval]:
            report = QueryRetrieval(
                query_id=query.query_id, query=query.query, status="no_results"
            )
            async with semaphore:
                try:
                    raw_results = await asyncio.to_thread(
                        self._retriever, query=query.query, top_k=self._top_k, law_name=None
                    )
                except Exception:
                    report.status = "failed"
                    report.error_code = "retrieval_failed"
                    return [], report
            if not isinstance(raw_results, list):
                report.status = "failed"
                report.error_code = "invalid_retrieval_response"
                return [], report
            laws: list[_RetrievedLaw] = []
            for result in raw_results:
                try:
                    laws.append(_RetrievedLaw.model_validate(result))
                except ValidationError:
                    report.invalid_result_count += 1
            if report.invalid_result_count:
                report.status = "partial" if laws else "failed"
                report.error_code = "invalid_retrieval_result"
            elif laws:
                report.status = "retrieved"
            return laws, report

        results = await asyncio.gather(*(search(query) for query in queries))
        registry: dict[str, ReviewSource] = {}
        reports: list[QueryRetrieval] = []
        # Merge in plan order, independent of query completion order.
        for laws, report in results:
            for law in laws:
                source_id = source_id_for(law.metadata.law_name, law.metadata.section, law.text)
                source = registry.setdefault(
                    source_id,
                    ReviewSource(
                        source_id=source_id,
                        law_name=law.metadata.law_name,
                        section=law.metadata.section,
                        text=law.text,
                    ),
                )
                if law.metadata.source_file and law.metadata.source_file not in source.source_files:
                    source.source_files.append(law.metadata.source_file)
                existing = next(
                    (match for match in source.matches if match.query_id == report.query_id), None
                )
                if existing is None:
                    source.matches.append(SourceMatch(query_id=report.query_id, score=law.score))
                elif law.score is not None and (existing.score is None or law.score > existing.score):
                    existing.score = law.score
                if source_id not in report.source_ids:
                    report.source_ids.append(source_id)
            reports.append(report)
        return RetrievalBundle(sources=list(registry.values()), queries=reports)

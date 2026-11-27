import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.llm.config import QueryRewriteWorkflow
from app.rag.query import retrieve_laws
from app.rag.rewrite import QueryRewriteResult, QueryRewriteService
from app.reviews.models import (
    ClauseQuery,
    ClauseRetrievalResult,
    ContractClause,
    QueryRetrievalResult,
    RetrievedLegalSource,
    RetrievalStatus,
    SourceMatch,
)


Retriever = Callable[..., list[dict[str, Any]]]


@dataclass(frozen=True)
class _QueryOutcome:
    query: ClauseQuery
    rewrite: QueryRewriteResult
    raw_results: list[dict[str, Any]] | None = None
    error_code: str | None = None


class ClauseRetriever:
    """Run the existing legal retriever without coupling to its internals."""

    def __init__(
        self,
        retriever: Retriever = retrieve_laws,
        query_rewriter: QueryRewriteService | None = None,
        *,
        max_concurrency: int = 1,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._retriever = retriever
        self._query_rewriter = query_rewriter
        self._max_concurrency = max_concurrency

    async def retrieve(
        self,
        clauses: list[ContractClause],
        queries: list[ClauseQuery],
        *,
        top_k: int = 5,
        rewrite_queries: bool = True,
        workflow: QueryRewriteWorkflow = QueryRewriteWorkflow.CONTRACT_REVIEW,
    ) -> list[ClauseRetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def run_query(query: ClauseQuery) -> _QueryOutcome:
            rewrite = (
                await self._query_rewriter.rewrite(
                    query.text,
                    workflow=workflow,
                )
                if rewrite_queries and self._query_rewriter is not None
                else QueryRewriteResult.disabled(query.text)
            )
            async with semaphore:
                try:
                    raw_results = await asyncio.to_thread(
                        self._retriever,
                        query=rewrite.effective_query,
                        top_k=top_k,
                        law_name=None,
                    )
                except Exception as error:  # isolate one retrieval from other clauses
                    return _QueryOutcome(
                        query=query,
                        rewrite=rewrite,
                        error_code=type(error).__name__,
                    )
                return _QueryOutcome(
                    query=query,
                    rewrite=rewrite,
                    raw_results=raw_results,
                )

        outcomes = await asyncio.gather(*(run_query(query) for query in queries))
        outcomes_by_clause: dict[str, list[_QueryOutcome]] = {}
        for outcome in outcomes:
            outcomes_by_clause.setdefault(outcome.query.clause_id, []).append(outcome)

        return [
            self._build_clause_result(
                clause,
                outcomes_by_clause.get(clause.clause_id, []),
            )
            for clause in clauses
        ]

    async def retrieve_additional(
        self,
        initial: ClauseRetrievalResult,
        query: ClauseQuery,
        *,
        top_k: int,
        law_name: str | None,
    ) -> tuple[ClauseRetrievalResult, bool]:
        rewrite = QueryRewriteResult.disabled(query.text)
        used_unfiltered_fallback = False
        try:
            raw_results = await asyncio.to_thread(
                self._retriever, query=query.text, top_k=top_k, law_name=law_name
            )
            if law_name and not any(
                self._parse_source(hit, query_id=query.query_id, rank=rank)
                for rank, hit in enumerate(raw_results, start=1)
            ):
                used_unfiltered_fallback = True
                raw_results = await asyncio.to_thread(
                    self._retriever, query=query.text, top_k=top_k, law_name=None
                )
            outcome = _QueryOutcome(query, rewrite, raw_results=raw_results)
        except Exception as error:
            outcome = _QueryOutcome(query, rewrite, error_code=type(error).__name__)

        updated = self._build_clause_result(initial.clause, [outcome], initial=initial)
        return updated, used_unfiltered_fallback

    def _build_clause_result(
        self,
        clause: ContractClause,
        outcomes: list[_QueryOutcome],
        *,
        initial: ClauseRetrievalResult | None = None,
    ) -> ClauseRetrievalResult:
        if not outcomes:
            raise ValueError(f"Clause {clause.clause_id} has no retrieval outcome")

        sources_by_id = {
            source.source_id: source.model_copy(deep=True)
            for source in (initial.sources if initial else [])
        }
        query_results = list(initial.query_results) if initial else []

        for outcome in outcomes:
            if outcome.error_code is not None:
                query_results.append(
                    QueryRetrievalResult(
                        query_id=outcome.query.query_id,
                        status=RetrievalStatus.FAILED,
                        error_code=outcome.error_code,
                        rewrite=outcome.rewrite,
                    )
                )
                continue

            source_ids: list[str] = []
            invalid_result_count = 0
            for rank, raw_result in enumerate(outcome.raw_results or [], start=1):
                parsed = self._parse_source(
                    raw_result,
                    query_id=outcome.query.query_id,
                    rank=rank,
                )
                if parsed is None:
                    invalid_result_count += 1
                    continue

                existing = sources_by_id.get(parsed.source_id)
                if existing is None:
                    sources_by_id[parsed.source_id] = parsed
                else:
                    existing.matches.extend(parsed.matches)
                    existing.matched_query_ids = list(
                        dict.fromkeys(
                            [*existing.matched_query_ids, *parsed.matched_query_ids]
                        )
                    )
                    existing.best_rank = min(existing.best_rank, parsed.best_rank)
                    scores = [
                        score
                        for score in (existing.best_score, parsed.best_score)
                        if score is not None
                    ]
                    existing.best_score = max(scores) if scores else None
                source_ids.append(parsed.source_id)

            unique_source_ids = list(dict.fromkeys(source_ids))
            query_results.append(
                QueryRetrievalResult(
                    query_id=outcome.query.query_id,
                    status=(
                        RetrievalStatus.READY
                        if unique_source_ids
                        else RetrievalStatus.NO_RESULTS
                    ),
                    source_ids=unique_source_ids,
                    invalid_result_count=invalid_result_count,
                    rewrite=outcome.rewrite,
                )
            )

        status = self._clause_status(query_results)
        sources = sorted(
            sources_by_id.values(),
            key=lambda source: (
                source.best_rank,
                -(source.best_score if source.best_score is not None else -1.0),
                source.source_id,
            ),
        )
        return ClauseRetrievalResult(
            clause=clause,
            queries=[
                *(initial.queries if initial else []),
                *(outcome.query for outcome in outcomes),
            ],
            query_results=query_results,
            sources=sources,
            status=status,
        )

    @staticmethod
    def _parse_source(
        raw_result: dict[str, Any],
        *,
        query_id: str,
        rank: int,
    ) -> RetrievedLegalSource | None:
        metadata = raw_result.get("metadata")
        text = raw_result.get("text")
        if not isinstance(metadata, dict) or not isinstance(text, str) or not text.strip():
            return None

        law_name = metadata.get("law_name")
        section = metadata.get("section")
        if not isinstance(law_name, str) or not law_name.strip():
            return None
        if not isinstance(section, str) or not section.strip():
            return None

        score_value = raw_result.get("score")
        score = float(score_value) if isinstance(score_value, (int, float)) else None
        clean_text = text.strip()
        identity = json.dumps(
            [law_name.strip(), section.strip(), clean_text],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        source_id = f"law:{hashlib.sha256(identity).hexdigest()[:16]}"
        match = SourceMatch(
            query_id=query_id,
            rank=rank,
            score=score,
            retrieval_score=ClauseRetriever._optional_float(
                raw_result.get("retrieval_score")
            ),
            rerank_score=ClauseRetriever._optional_float(
                raw_result.get("rerank_score")
            ),
            retrieval_rank=ClauseRetriever._optional_positive_int(
                raw_result.get("retrieval_rank")
            ),
            rerank_rank=ClauseRetriever._optional_positive_int(
                raw_result.get("rerank_rank")
            ),
        )
        source_file = metadata.get("source_file")

        return RetrievedLegalSource(
            source_id=source_id,
            law_name=law_name.strip(),
            section=section.strip(),
            text=clean_text,
            source_file=(source_file if isinstance(source_file, str) else None),
            best_score=score,
            best_rank=rank,
            matched_query_ids=[query_id],
            matches=[match],
        )

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        return float(value) if isinstance(value, (int, float)) else None

    @staticmethod
    def _optional_positive_int(value: Any) -> int | None:
        return value if isinstance(value, int) and value >= 1 else None

    @staticmethod
    def _clause_status(
        query_results: list[QueryRetrievalResult],
    ) -> RetrievalStatus:
        statuses = [result.status for result in query_results]
        if all(status == RetrievalStatus.READY for status in statuses):
            return RetrievalStatus.READY
        if any(status == RetrievalStatus.READY for status in statuses):
            return RetrievalStatus.PARTIAL
        if all(status == RetrievalStatus.NO_RESULTS for status in statuses):
            return RetrievalStatus.NO_RESULTS
        return RetrievalStatus.FAILED


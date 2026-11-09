from app.documents.models import ParsedDocument
from app.rag.query import retrieve_laws
from app.reviews.models import (
    ClauseRetrievalPreparation,
    PreparationStatus,
    PreparationSummary,
    RetrievalStatus,
)
from app.reviews.queries import ClauseQueryBuilder
from app.reviews.retrieval import ClauseRetriever, Retriever
from app.reviews.segmentation import segment_contract


class ClauseRetrievalService:
    def __init__(
        self,
        retriever: Retriever = retrieve_laws,
        *,
        max_concurrency: int = 1,
        max_query_chars: int = 4000,
    ) -> None:
        self._query_builder = ClauseQueryBuilder(
            max_query_chars=max_query_chars
        )
        self._retriever = ClauseRetriever(
            retriever,
            max_concurrency=max_concurrency,
        )

    async def prepare(
        self,
        document: ParsedDocument,
        *,
        top_k: int = 5,
    ) -> ClauseRetrievalPreparation:
        clauses = segment_contract(document.blocks)
        queries = [
            query
            for clause in clauses
            for query in self._query_builder.build(clause, document.blocks)
        ]
        clause_results = await self._retriever.retrieve(
            clauses,
            queries,
            top_k=top_k,
        )

        counts = {
            status: sum(result.status == status for result in clause_results)
            for status in RetrievalStatus
        }
        if counts[RetrievalStatus.READY] == len(clause_results):
            preparation_status = PreparationStatus.READY
        elif counts[RetrievalStatus.READY] or counts[RetrievalStatus.PARTIAL]:
            preparation_status = PreparationStatus.PARTIAL
        else:
            preparation_status = PreparationStatus.FAILED

        return ClauseRetrievalPreparation(
            document_id=document.document_id,
            filename=document.filename,
            status=preparation_status,
            clause_results=clause_results,
            summary=PreparationSummary(
                block_count=len(document.blocks),
                clause_count=len(clauses),
                query_count=len(queries),
                retrieved_source_count=sum(
                    len(result.sources) for result in clause_results
                ),
                ready_clause_count=counts[RetrievalStatus.READY],
                partial_clause_count=counts[RetrievalStatus.PARTIAL],
                no_results_clause_count=counts[RetrievalStatus.NO_RESULTS],
                failed_clause_count=counts[RetrievalStatus.FAILED],
            ),
        )


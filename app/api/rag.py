import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.llm.config import QueryRewriteWorkflow
from app.rag.query import retrieve_laws
from app.rag.rewrite import QueryRewriteResult, QueryRewriteService



router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


class RetrieveRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Thai legal question or contract clause",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of law chunks to retrieve",
    )
    law_name: str | None = Field(
        default=None,
        description="Optional law name metadata filter",
    )


class RetrieveResult(BaseModel):
    score: float | None
    retrieval_score: float | None = None
    rerank_score: float | None = None
    retrieval_rank: int | None = None
    rerank_rank: int | None = None
    text: str
    metadata: dict[str, Any]


class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrieveResult]
    rewrite: QueryRewriteResult


def get_query_rewrite_service(request: Request) -> QueryRewriteService:
    service = getattr(request.app.state, "query_rewrite_service", None)
    if not isinstance(service, QueryRewriteService):
        raise RuntimeError("Query rewrite service is not initialized")
    return service


QueryRewriteDependency = Annotated[
    QueryRewriteService,
    Depends(get_query_rewrite_service),
]


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    request: RetrieveRequest,
    query_rewriter: QueryRewriteDependency,
) -> RetrieveResponse:
    try:
        rewrite = await query_rewriter.rewrite(
            request.query,
            workflow=QueryRewriteWorkflow.RAG,
        )
        results = await asyncio.to_thread(
            retrieve_laws,
            query=rewrite.effective_query,
            top_k=request.top_k,
            law_name=request.law_name,
        )

        return RetrieveResponse(
            query=request.query,
            results=results,
            rewrite=rewrite,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error
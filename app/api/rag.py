from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.rag.query import retrieve_laws


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
    topic: str | None = Field(
        default=None,
        description="Optional legal topic metadata filter",
    )
    law_name: str | None = Field(
        default=None,
        description="Optional law name metadata filter",
    )


class RetrieveResult(BaseModel):
    score: float | None
    text: str
    metadata: dict[str, Any]


class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrieveResult]


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    try:
        results = retrieve_laws(
            query=request.query,
            top_k=request.top_k,
            topic=request.topic,
            law_name=request.law_name,
        )

        return RetrieveResponse(
            query=request.query,
            results=results,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error
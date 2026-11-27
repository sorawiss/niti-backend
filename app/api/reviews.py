import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from pydantic import Field

from app.api.uploads import parse_uploaded_docx
from app.documents.models import DocumentBlock
from app.llm.config import QueryRewriteWorkflow
from app.reviews.review_models import ContractReviewResult
from app.reviews.reviewer import ClauseReviewService
from app.reviews.segmentation import ClauseSegmentationError
from app.reviews.service import ClauseRetrievalService
from app.reviews.whole_contract_models import WholeContractReviewPublicResult
from app.reviews.whole_contract_reviewer import WholeContractReviewService


logger = logging.getLogger("uvicorn.error")
router = APIRouter(prefix="/reviews", tags=["Reviews"])


class ContractReviewApiResponse(ContractReviewResult):
    document_blocks: list[DocumentBlock]


class WholeContractReviewApiResponse(WholeContractReviewPublicResult):
    document_blocks: list[DocumentBlock]
    workflow_debug: dict[str, Any] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


def get_clause_retrieval_service(request: Request) -> ClauseRetrievalService:
    service = getattr(request.app.state, "clause_retrieval_service", None)
    if not isinstance(service, ClauseRetrievalService):
        raise RuntimeError("Clause retrieval service is not initialized")
    return service


def get_clause_review_service(request: Request) -> ClauseReviewService:
    service = getattr(request.app.state, "clause_review_service", None)
    if not isinstance(service, ClauseReviewService):
        raise RuntimeError("Clause review service is not initialized")
    return service


def get_whole_contract_review_service(
    request: Request,
) -> WholeContractReviewService:
    service = getattr(request.app.state, "whole_contract_review_service", None)
    if not isinstance(service, WholeContractReviewService):
        raise RuntimeError("Whole-contract review service is not initialized")
    return service


RetrievalServiceDependency = Annotated[
    ClauseRetrievalService,
    Depends(get_clause_retrieval_service),
]
ReviewServiceDependency = Annotated[
    ClauseReviewService,
    Depends(get_clause_review_service),
]
WholeContractReviewServiceDependency = Annotated[
    WholeContractReviewService,
    Depends(get_whole_contract_review_service),
]


@router.post("", response_model=ContractReviewApiResponse)
async def review_document(
    file: Annotated[
        UploadFile,
        File(description="A DOCX document no larger than 5 MiB."),
    ],
    retrieval_service: RetrievalServiceDependency,
    review_service: ReviewServiceDependency,
) -> ContractReviewApiResponse:
    document = await parse_uploaded_docx(file)
    try:
        preparation = await retrieval_service.prepare(
            document,
            top_k=5,
            workflow=QueryRewriteWorkflow.CONTRACT_REVIEW,
        )
        review = await review_service.review(preparation)
        return ContractReviewApiResponse(
            **review.model_dump(mode="python"),
            document_blocks=document.blocks,
        )
    except ClauseSegmentationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Document contains no reviewable text",
        ) from error
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "review_workflow_failed",
                "message": "ไม่สามารถประมวลผลการตรวจสัญญาได้",
                "retryable": False,
            },
        ) from error


@router.post(
    "/whole-contract",
    response_model=WholeContractReviewApiResponse,
)
async def review_whole_contract(
    file: Annotated[
        UploadFile,
        File(description="A DOCX document no larger than 5 MiB."),
    ],
    retrieval_service: RetrievalServiceDependency,
    review_service: WholeContractReviewServiceDependency,
    debug: bool = False,
) -> WholeContractReviewApiResponse:
    document = await parse_uploaded_docx(file)
    try:
        preparation = await retrieval_service.prepare(
            document,
            top_k=5,
            workflow=QueryRewriteWorkflow.WHOLE_CONTRACT,
            rewrite_queries=False,
        )
        review = await review_service.review(preparation, debug=debug)
        if debug and review.workflow_debug is not None and logger.isEnabledFor(logging.INFO):
            trace = review.model_dump(
                mode="json", include={"document_id", "workflow_debug"}
            )
            logger.info(
                "whole_contract_workflow_debug=%s",
                json.dumps(trace, ensure_ascii=False, separators=(",", ":")),
            )
        return WholeContractReviewApiResponse(
            **review.model_dump(mode="python", exclude={"llm_debug"}),
            document_blocks=document.blocks,
        )
    except ClauseSegmentationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Document contains no reviewable text",
        ) from error
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "whole_contract_review_workflow_failed",
                "message": "Unable to process the whole-contract review",
                "retryable": False,
            },
        ) from error

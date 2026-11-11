from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.api.uploads import parse_uploaded_docx
from app.documents.models import DocumentBlock
from app.reviews.review_models import ContractReviewResult
from app.reviews.reviewer import ClauseReviewService
from app.reviews.segmentation import ClauseSegmentationError
from app.reviews.service import ClauseRetrievalService


router = APIRouter(prefix="/reviews", tags=["Reviews"])


class ContractReviewApiResponse(ContractReviewResult):
    document_blocks: list[DocumentBlock]


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


RetrievalServiceDependency = Annotated[
    ClauseRetrievalService,
    Depends(get_clause_retrieval_service),
]
ReviewServiceDependency = Annotated[
    ClauseReviewService,
    Depends(get_clause_review_service),
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
        preparation = await retrieval_service.prepare(document, top_k=5)
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
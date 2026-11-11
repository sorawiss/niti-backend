from typing import Annotated

from fastapi import APIRouter, File, UploadFile

from app.api.uploads import MAX_FILE_SIZE, parse_uploaded_docx
from app.documents.models import ParsedDocument


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post("/upload", response_model=ParsedDocument)
async def upload_document(
    file: Annotated[
        UploadFile,
        File(description="A DOCX document no larger than 5 MiB."),
    ],
) -> ParsedDocument:
    """Validate an uploaded DOCX file and return its ordered content blocks."""

    return await parse_uploaded_docx(file)

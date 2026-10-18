from typing import Annotated
from uuid import uuid4
from zipfile import BadZipFile

from docx.opc.exceptions import PackageNotFoundError
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from lxml.etree import XMLSyntaxError

from app.documents.models import ParsedDocument
from app.documents.parser import parse_docx


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

MAX_FILE_SIZE = 5 * 1024 * 1024


@router.post("/upload", response_model=ParsedDocument)
async def upload_document(
    file: Annotated[
        UploadFile,
        File(description="A DOCX document no larger than 5 MiB."),
    ],
) -> ParsedDocument:
    """Validate an uploaded DOCX file and return its ordered content blocks."""

    filename = file.filename or ""
    if not filename.lower().endswith(".docx"):
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .docx files are supported",
        )

    try:
        content = await file.read(MAX_FILE_SIZE + 1)
    finally:
        await file.close()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="DOCX file is empty",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="File exceeds the 5 MiB limit",
        )

    try:
        blocks = parse_docx(content)
    except (
        BadZipFile,
        PackageNotFoundError,
        XMLSyntaxError,
        KeyError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid DOCX document",
        ) from error

    return ParsedDocument(
        document_id=str(uuid4()),
        filename=filename,
        blocks=blocks,
    )

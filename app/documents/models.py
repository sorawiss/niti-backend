from typing import Literal

from pydantic import BaseModel, Field


class DocumentBlock(BaseModel):
    """One top-level content block extracted from a document."""

    index: int
    type: Literal["heading", "paragraph", "table"]
    text: str
    style: str | None = None


class ParsedDocument(BaseModel):
    """Structured representation returned for an uploaded document."""

    document_id: str
    filename: str
    blocks: list[DocumentBlock] = Field(default_factory=list)

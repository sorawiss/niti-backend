from io import BytesIO

from docx import Document
from docx.document import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.documents.models import DocumentBlock


def parse_docx(content: bytes) -> list[DocumentBlock]:
    """Extract top-level paragraphs and tables in their original order."""

    document: DocxDocument = Document(BytesIO(content))
    blocks: list[DocumentBlock] = []

    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue

            style = item.style.name if item.style else None
            block_type = (
                "heading"
                if style is not None and style.startswith("Heading")
                else "paragraph"
            )
        elif isinstance(item, Table):
            rows = [
                " | ".join(cell.text.strip() for cell in row.cells)
                for row in item.rows
            ]
            text = "\n".join(rows)
            style = None
            block_type = "table"
        else:
            continue

        blocks.append(
            DocumentBlock(
                index=len(blocks),
                type=block_type,
                text=text,
                style=style,
            )
        )

    return blocks

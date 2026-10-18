from io import BytesIO
import unittest
from uuid import UUID

from docx import Document
from fastapi.testclient import TestClient

from app.api.documents import MAX_FILE_SIZE
from app.main import app


client = TestClient(app)


def make_docx(text: str) -> bytes:
    document = Document()
    document.add_paragraph(text)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


class DocumentUploadTests(unittest.TestCase):
    def test_uploads_docx_and_returns_structured_json(self) -> None:
        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "contract.docx",
                    make_docx("สัญญาจ้างแรงงาน"),
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document",
                )
            },
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        UUID(result["document_id"])
        self.assertEqual(result["filename"], "contract.docx")
        self.assertEqual(result["blocks"][0]["index"], 0)
        self.assertEqual(result["blocks"][0]["type"], "paragraph")
        self.assertEqual(result["blocks"][0]["text"], "สัญญาจ้างแรงงาน")

    def test_rejects_unsupported_extension(self) -> None:
        response = client.post(
            "/documents/upload",
            files={"file": ("contract.txt", b"plain text", "text/plain")},
        )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["detail"], "Only .docx files are supported")

    def test_rejects_empty_upload(self) -> None:
        response = client.post(
            "/documents/upload",
            files={"file": ("contract.docx", b"", "application/octet-stream")},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "DOCX file is empty")

    def test_rejects_invalid_docx_package(self) -> None:
        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "contract.docx",
                    b"not a DOCX package",
                    "application/octet-stream",
                )
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "Invalid DOCX document")

    def test_rejects_oversized_upload(self) -> None:
        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "contract.docx",
                    b"x" * (MAX_FILE_SIZE + 1),
                    "application/octet-stream",
                )
            },
        )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"], "File exceeds the 5 MiB limit")


if __name__ == "__main__":
    unittest.main()

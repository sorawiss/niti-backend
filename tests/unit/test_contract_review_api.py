from io import BytesIO
import json
import unittest

from docx import Document
from fastapi.testclient import TestClient

from app.api.reviews import (
    get_clause_retrieval_service,
    get_clause_review_service,
)
from app.api.uploads import MAX_FILE_SIZE
from app.documents.models import ParsedDocument
from app.main import app
from app.reviews.models import (
    BoundaryReason,
    ContractClause,
    RetrievalStatus,
)
from app.reviews.review_models import (
    AssessmentConfidence,
    ClauseReviewResult,
    ClauseReviewStatus,
    ContractReviewResult,
    LegalAssessment,
    LegalAssessmentOutcome,
    LegalCitation,
    ReviewRunStatus,
    ReviewSummary,
)
from app.reviews.segmentation import ClauseSegmentationError


client = TestClient(app)


def make_docx(text: str | None = "1. Working hours") -> bytes:
    document = Document()
    if text is not None:
        document.add_paragraph(text)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_structured_docx() -> bytes:
    document = Document()
    document.add_heading("Employment contract", level=1)
    document.add_paragraph("1. Working hours")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Day"
    table.cell(0, 1).text = "Hours"
    table.cell(1, 0).text = "Monday"
    table.cell(1, 1).text = "8"
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_review(document: ParsedDocument, status: ReviewRunStatus) -> ContractReviewResult:
    clause = ContractClause(
        clause_id="clause-0001",
        order=0,
        number="1",
        title="Working hours",
        text="1. Working hours",
        block_indices=[0],
        section_path=["Working hours"],
        boundary_reason=BoundaryReason.NUMBERED_CLAUSE,
    )
    citation = LegalCitation(
        source_id="law:hours",
        law_name="Labor Protection Act",
        section="Section 23",
        text="Working hours must comply with the statutory maximum.",
        source_file="labor-law.jsonl",
    )
    assessment = LegalAssessment(
        assessment_id="assessment-01",
        outcome=LegalAssessmentOutcome.COMPLIANT,
        issue="เวลาทำงานปกติ",
        reasoning="ข้อสัญญาสอดคล้องกับหลักฐานกฎหมายที่อ้างอิง",
        citations=[citation],
        confidence=AssessmentConfidence.HIGH,
    )
    clause_status = (
        ClauseReviewStatus.REVIEW_FAILED
        if status == ReviewRunStatus.FAILED
        else ClauseReviewStatus.NO_ISSUE_FOUND
    )
    assessments = [] if clause_status == ClauseReviewStatus.REVIEW_FAILED else [assessment]
    return ContractReviewResult(
        document_id=document.document_id,
        filename=document.filename,
        status=status,
        clause_reviews=[
            ClauseReviewResult(
                clause=clause,
                retrieval_status=RetrievalStatus.READY,
                status=clause_status,
                applicability_reason="ข้อสัญญากำหนดเงื่อนไขการจ้างงาน",
                assessments=assessments,
                error_code=(
                    "invalid_model_output"
                    if clause_status == ClauseReviewStatus.REVIEW_FAILED
                    else None
                ),
            )
        ],
        summary=ReviewSummary(
            clause_count=1,
            assessment_count=len(assessments),
            compliant_assessment_count=len(assessments),
            potential_violation_assessment_count=0,
            needs_more_evidence_assessment_count=0,
            no_issue_found_count=(
                1 if clause_status == ClauseReviewStatus.NO_ISSUE_FOUND else 0
            ),
            potential_violation_count=0,
            needs_more_evidence_count=0,
            not_applicable_count=0,
            review_failed_count=(
                1 if clause_status == ClauseReviewStatus.REVIEW_FAILED else 0
            ),
        ),
    )


class FakeRetrievalService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.top_k: int | None = None

    async def prepare(
        self,
        document: ParsedDocument,
        *,
        top_k: int = 5,
    ) -> ParsedDocument:
        self.top_k = top_k
        if self.error is not None:
            raise self.error
        return document


class FakeReviewService:
    def __init__(self, status: ReviewRunStatus = ReviewRunStatus.READY) -> None:
        self.status = status

    async def review(self, document: ParsedDocument) -> ContractReviewResult:
        return make_review(document, self.status)


class ContractReviewApiTests(unittest.TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def post_review(
        self,
        *,
        filename: str = "contract.docx",
        content: bytes | None = None,
        retrieval: FakeRetrievalService | None = None,
        review: FakeReviewService | None = None,
    ):
        retrieval_service = retrieval or FakeRetrievalService()
        review_service = review or FakeReviewService()
        app.dependency_overrides[get_clause_retrieval_service] = (
            lambda: retrieval_service
        )
        app.dependency_overrides[get_clause_review_service] = lambda: review_service
        return client.post(
            "/reviews",
            files={
                "file": (
                    filename,
                    make_docx() if content is None else content,
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document",
                )
            },
        )

    def test_valid_docx_returns_clause_review_without_excerpt(self) -> None:
        retrieval = FakeRetrievalService()

        response = self.post_review(retrieval=retrieval)

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["clause_reviews"][0]["clause"]["number"], "1")
        self.assertEqual(retrieval.top_k, 5)
        self.assertNotIn("contract_excerpt", json.dumps(result))

    def test_document_blocks_preserve_parser_order_type_text_and_style(self) -> None:
        response = self.post_review(content=make_structured_docx())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["document_blocks"],
            [
                {
                    "index": 0,
                    "type": "heading",
                    "text": "Employment contract",
                    "style": "Heading 1",
                },
                {
                    "index": 1,
                    "type": "paragraph",
                    "text": "1. Working hours",
                    "style": "Normal",
                },
                {
                    "index": 2,
                    "type": "table",
                    "text": "Day | Hours\nMonday | 8",
                    "style": None,
                },
            ],
        )

    def test_partial_and_failed_reviews_are_successful_http_responses(self) -> None:
        for run_status in (ReviewRunStatus.PARTIAL, ReviewRunStatus.FAILED):
            with self.subTest(run_status=run_status):
                response = self.post_review(
                    review=FakeReviewService(run_status),
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], run_status.value)

    def test_rejects_unsupported_extension(self) -> None:
        response = self.post_review(filename="contract.txt", content=b"text")
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["detail"], "Only .docx files are supported")

    def test_rejects_empty_upload(self) -> None:
        response = self.post_review(content=b"")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "DOCX file is empty")

    def test_rejects_oversized_upload(self) -> None:
        response = self.post_review(content=b"x" * (MAX_FILE_SIZE + 1))
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"], "File exceeds the 5 MiB limit")

    def test_rejects_invalid_docx(self) -> None:
        response = self.post_review(content=b"not a docx")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "Invalid DOCX document")

    def test_rejects_document_without_reviewable_text(self) -> None:
        retrieval = FakeRetrievalService(
            ClauseSegmentationError("The document contains no reviewable text")
        )
        response = self.post_review(content=make_docx(None), retrieval=retrieval)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["detail"],
            "Document contains no reviewable text",
        )


if __name__ == "__main__":
    unittest.main()

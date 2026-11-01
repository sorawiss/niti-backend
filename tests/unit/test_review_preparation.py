import json
import threading
import unittest

from app.documents.models import DocumentBlock, ParsedDocument
from app.llm.base import LLMGateway
from app.llm.models import LLMRequest, LLMResponse
from app.reviews.models import ContextDisposition, ReviewPlan
from app.reviews.planner import (
    ReviewInputError,
    ReviewPlanner,
    ReviewPlanningError,
    validate_plan,
)
from app.reviews.prompts import BASELINE_TOPICS
from app.reviews.retrieval import ReviewRetriever, source_id_for
from app.reviews.service import ReviewPreparationService


def document() -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-1",
        filename="contract.docx",
        blocks=[
            DocumentBlock(index=0, type="paragraph", text="ค่าจ้างเดือนละ 45,000 บาท"),
            DocumentBlock(index=1, type="table", text="เวลางาน | 09.00–18.00\nพัก | 12.00–13.00"),
            DocumentBlock(index=2, type="paragraph", text="ลงชื่อ นายตัวอย่าง"),
        ],
    )


def plan() -> ReviewPlan:
    return ReviewPlan.model_validate({
        "topics": [
            {
                "topic_id": topic_id,
                "description": description,
                "block_indices": [0, 1] if index == 0 else [],
                "queries": [{"query_id": f"q{index}", "query": description}],
            }
            for index, (topic_id, description) in enumerate(BASELINE_TOPICS.items())
        ],
        "context_blocks": [{"block_index": 2, "reason": "ลายเซ็น"}],
    })


def law(*, text="ข้อความกฎหมาย", section="มาตรา 24", score=0.55, source_file="chap2.jsonl"):
    return {
        "text": text,
        "score": score,
        "metadata": {
            "law_name": "พระราชบัญญัติคุ้มครองแรงงาน พ.ศ. 2541",
            "section": section,
            "source_file": source_file,
        },
    }


class FakeGateway(LLMGateway):
    def __init__(self, content: str | None = None, *, finish_reason="stop"):
        self.content = content if content is not None else plan().model_dump_json()
        self.finish_reason = finish_reason
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            content=self.content, model="test/model", provider="fake",
            finish_reason=self.finish_reason,
        )


class ReviewPlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_entire_document_and_strict_schema(self):
        gateway = FakeGateway()
        result, response = await ReviewPlanner(gateway).plan(document())
        request = gateway.requests[0]
        serialized = json.loads(request.messages[1].content.removeprefix("CONTRACT\n"))
        self.assertEqual(serialized, document().model_dump())
        self.assertEqual(request.max_output_tokens, 8000)
        self.assertTrue(request.response_format["json_schema"]["strict"])
        self.assertFalse(request.response_format["json_schema"]["schema"]["additionalProperties"])
        self.assertEqual(len(result.topics), 4)
        self.assertEqual(response.model, "test/model")

    async def test_invalid_json_and_truncated_or_refused_response_fail(self):
        for gateway in [FakeGateway("not JSON"), FakeGateway(finish_reason="length"),
                        FakeGateway(finish_reason="content_filter")]:
            with self.subTest(finish_reason=gateway.finish_reason, content=gateway.content):
                with self.assertRaises(ReviewPlanningError):
                    await ReviewPlanner(gateway).plan(document())

    async def test_oversized_or_invalid_document_skips_llm(self):
        for change in ["empty", "duplicate", "negative", "oversized"]:
            with self.subTest(change=change):
                doc = document()
                if change == "empty":
                    doc.blocks = []
                elif change == "duplicate":
                    doc.blocks[1].index = 0
                elif change == "negative":
                    doc.blocks[1].index = -1
                gateway = FakeGateway()
                planner = ReviewPlanner(gateway, max_input_bytes=1 if change == "oversized" else 60000)
                with self.assertRaises(ReviewInputError):
                    await planner.plan(doc)
                self.assertEqual(gateway.requests, [])

    def test_rejects_invalid_plan_references_and_incomplete_coverage(self):
        for change in ["unknown", "uncovered", "duplicate_topic", "duplicate_query",
                       "missing_baseline", "context_overlap", "duplicate_context", "too_many_queries"]:
            with self.subTest(change=change):
                candidate = plan()
                if change == "unknown":
                    candidate.topics[0].block_indices.append(99)
                elif change == "uncovered":
                    candidate.topics[0].block_indices = [0]
                elif change == "duplicate_topic":
                    candidate.topics.append(candidate.topics[0].model_copy(deep=True))
                elif change == "duplicate_query":
                    candidate.topics[1].queries[0].query_id = "q0"
                elif change == "missing_baseline":
                    candidate.topics.pop()
                elif change == "context_overlap":
                    candidate.context_blocks.append(ContextDisposition(block_index=0, reason="context"))
                elif change == "duplicate_context":
                    candidate.context_blocks.append(candidate.context_blocks[0])
                with self.assertRaises(ReviewPlanningError):
                    validate_plan(document(), candidate, max_queries=3 if change == "too_many_queries" else 32)

    def test_allows_cross_clause_topics_and_empty_baseline_references(self):
        candidate = plan()
        candidate.topics[1].block_indices = [0, 1]
        validate_plan(document(), candidate)


class ReviewRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_deduplicates_sources_and_preserves_query_provenance(self):
        calls = []

        def retrieve(**kwargs):
            calls.append(kwargs)
            return [law(score=0.2), law(score=0.8, source_file="copy.jsonl")]

        result = await ReviewRetriever(retrieve).retrieve(plan())
        self.assertEqual(len(result.sources), 1)
        source = result.sources[0]
        self.assertEqual(source.source_files, ["chap2.jsonl", "copy.jsonl"])
        self.assertEqual([match.query_id for match in source.matches], ["q0", "q1", "q2", "q3"])
        self.assertTrue(all(match.score == 0.8 for match in source.matches))
        self.assertTrue(all(report.source_ids == [source.source_id] for report in result.queries))
        self.assertTrue(all(call["law_name"] is None and call["top_k"] == 5 for call in calls))

    def test_source_identity_preserves_provision_and_text_variants(self):
        original = source_id_for("law", "24", "text")
        self.assertEqual(original, source_id_for("law", "24", "text"))
        self.assertEqual(len({original, source_id_for("other", "24", "text"),
                              source_id_for("law", "25", "text"),
                              source_id_for("law", "24", "text ")}), 4)

    async def test_different_versions_of_same_section_remain_separate(self):
        result = await ReviewRetriever(lambda **_: [law(), law(text="different text")]).retrieve(plan())
        self.assertEqual(len(result.sources), 2)
        self.assertEqual(len(result.queries[0].source_ids), 2)

    async def test_empty_failed_and_malformed_results_are_explicit(self):
        def failed(**_):
            raise RuntimeError("private upstream details")

        cases = [
            (lambda **_: [], "no_results", None),
            (failed, "failed", "retrieval_failed"),
            (lambda **_: {}, "failed", "invalid_retrieval_response"),
            (lambda **_: [law(), {"text": "missing metadata"}], "partial", "invalid_retrieval_result"),
            (lambda **_: [law(score=float("nan")), law(text=" ")], "failed", "invalid_retrieval_result"),
        ]
        for retriever, status, code in cases:
            with self.subTest(status=status, code=code):
                result = await ReviewRetriever(retriever).retrieve(plan())
                self.assertTrue(all(report.status == status for report in result.queries))
                self.assertTrue(all(report.error_code == code for report in result.queries))
                self.assertNotIn("private upstream details", result.model_dump_json())

    async def test_bounds_concurrency_and_keeps_plan_order(self):
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        active = 0
        peak = 0

        def retrieve(**kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                barrier.wait(timeout=5)
                return [law(text=kwargs["query"])]
            finally:
                with lock:
                    active -= 1

        result = await ReviewRetriever(retrieve, concurrency=2).retrieve(plan())
        self.assertEqual(peak, 2)
        self.assertTrue(all(report.status == "retrieved" for report in result.queries))
        self.assertEqual([report.query_id for report in result.queries], ["q0", "q1", "q2", "q3"])

    async def test_limits_are_validated_before_retrieval(self):
        with self.assertRaises(ValueError):
            ReviewRetriever(top_k=0)
        with self.assertRaises(ValueError):
            await ReviewRetriever(max_queries=3).retrieve(plan())
        candidate = plan()
        candidate.topics[1].queries[0].query_id = "q0"
        with self.assertRaises(ValueError):
            await ReviewRetriever().retrieve(candidate)


class ReviewPreparationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_preparation_returns_plan_sources_and_metadata(self):
        service = ReviewPreparationService(
            ReviewPlanner(FakeGateway()), ReviewRetriever(lambda **_: [law()])
        )
        result = await service.prepare(document())
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.document_id, "doc-1")
        self.assertEqual(result.model, "test/model")
        self.assertEqual(len(result.retrieval.sources), 1)
        self.assertEqual(len(result.plan.topics), 4)

    async def test_no_results_are_partial_not_successful_legal_assessment(self):
        result = await ReviewPreparationService(
            ReviewPlanner(FakeGateway()), ReviewRetriever(lambda **_: [])
        ).prepare(document())
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.retrieval.sources, [])

    async def test_invalid_plan_never_reaches_retrieval(self):
        calls = []
        service = ReviewPreparationService(
            ReviewPlanner(FakeGateway("{}")),
            ReviewRetriever(lambda **kwargs: calls.append(kwargs)),
        )
        with self.assertRaises(ReviewPlanningError):
            await service.prepare(document())
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()

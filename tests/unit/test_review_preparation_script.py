import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

from docx import Document

from app.llm.errors import LLMTimeoutError
from app.reviews.retrieval import ReviewRetriever
from scripts.try_review_preparation import run
from tests.unit.test_review_preparation import FakeGateway, law, plan


class ReviewPreparationScriptTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.document_path = self.root / "mock.docx"
        self.output = self.root / "run"
        doc = Document()
        doc.add_paragraph("ค่าจ้างเดือนละ 45,000 บาท")
        doc.add_table(rows=1, cols=1).cell(0, 0).text = "เวลาทำงาน"
        doc.add_paragraph("ลงชื่อ นายตัวอย่าง")
        doc.save(self.document_path)
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.enterContext(redirect_stdout(self.stdout))
        self.enterContext(redirect_stderr(self.stderr))

    def read(self, filename):
        return json.loads((self.output / filename).read_text(encoding="utf-8"))

    async def test_success_preserves_document_plan_response_and_result(self):
        gateway = FakeGateway()
        gateway.close = AsyncMock()
        code = await run(
            self.document_path, self.output, gateway=gateway,
            retriever=ReviewRetriever(lambda **_: [law()]),
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(self.read("document.json")["blocks"]), 3)
        self.assertEqual(self.read("plan.json"), plan().model_dump())
        self.assertEqual(self.read("planner-response.json")["content"], gateway.content)
        self.assertEqual(self.read("result.json")["status"], "ready")
        self.assertEqual(self.read("run.json")["query_count"], 4)
        self.assertEqual(len(gateway.requests), 1)
        gateway.close.assert_not_awaited()

    async def test_rejected_budget_saves_count_and_plan_without_retrieval_or_retry(self):
        rejected = plan()
        for index, topic in enumerate(rejected.topics):
            original = topic.queries[0]
            topic.queries = [
                original.model_copy(update={"query_id": f"q{index}-{number}"})
                for number in range(9)
            ]
        gateway = FakeGateway(rejected.model_dump_json())
        calls = []
        code = await run(
            self.document_path, self.output, gateway=gateway,
            retriever=ReviewRetriever(lambda **kwargs: calls.append(kwargs)),
        )
        self.assertEqual(code, 1)
        diagnostics = self.read("run.json")
        self.assertEqual(diagnostics["stage"], "planning")
        self.assertEqual(diagnostics["reason"], "Review query limit exceeded")
        self.assertEqual(diagnostics["query_count"], 36)
        self.assertEqual(diagnostics["max_queries"], 32)
        self.assertEqual(self.read("rejected-plan.json"), rejected.model_dump())
        self.assertEqual(self.read("planner-response.json")["content"], gateway.content)
        self.assertFalse((self.output / "result.json").exists())
        self.assertEqual(len(gateway.requests), 1)
        self.assertEqual(calls, [])

    async def test_bad_json_schema_and_truncation_keep_raw_response(self):
        for index, gateway in enumerate([
            FakeGateway("not JSON"), FakeGateway("{}"), FakeGateway(finish_reason="length"),
        ]):
            with self.subTest(case=index):
                self.output = self.root / f"failed-{index}"
                code = await run(self.document_path, self.output, gateway=gateway)
                self.assertEqual(code, 1)
                self.assertEqual(self.read("planner-response.json")["content"], gateway.content)
                self.assertIsNone(self.read("run.json")["query_count"])
                self.assertFalse((self.output / "rejected-plan.json").exists())
                self.assertEqual(len(gateway.requests), 1)

    async def test_coverage_failure_keeps_rejected_plan(self):
        rejected = plan()
        rejected.topics[0].block_indices = [0]
        code = await run(
            self.document_path, self.output, gateway=FakeGateway(rejected.model_dump_json())
        )
        self.assertEqual(code, 1)
        self.assertIn("unaccounted", self.read("run.json")["reason"])
        self.assertEqual(self.read("run.json")["query_count"], 4)

    async def test_partial_retrieval_still_saves_accepted_plan_and_result(self):
        code = await run(
            self.document_path, self.output, gateway=FakeGateway(),
            retriever=ReviewRetriever(lambda **_: []),
        )
        self.assertEqual(code, 2)
        self.assertEqual(self.read("result.json")["status"], "partial")
        self.assertEqual(self.read("run.json")["stage"], "completed")
        self.assertTrue((self.output / "plan.json").is_file())

    async def test_missing_input_or_existing_output_does_not_call_gateway(self):
        gateway = FakeGateway()
        with self.assertRaises(FileNotFoundError):
            await run(self.root / "missing.docx", self.output, gateway=gateway)
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        marker = self.output / "previous.txt"
        marker.write_text("preserve me", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            await run(self.document_path, self.output, gateway=gateway)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve me")
        self.assertEqual(gateway.requests, [])

    async def test_owned_gateway_is_closed_and_error_details_are_not_logged(self):
        gateway = FakeGateway()
        gateway.generate = AsyncMock(side_effect=LLMTimeoutError(
            "private details", code="llm_timeout", retryable=True,
        ))
        gateway.close = AsyncMock()
        with patch("scripts.try_review_preparation.create_llm_gateway", return_value=gateway):
            code = await run(self.document_path, self.output)
        self.assertEqual(code, 1)
        self.assertEqual(self.read("run.json")["code"], "llm_timeout")
        self.assertNotIn("private details", self.stderr.getvalue())
        gateway.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()

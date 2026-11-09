import unittest

from app.documents.models import DocumentBlock, ParsedDocument
from app.reviews.models import (
    BoundaryReason,
    ClauseQuery,
    ContractClause,
    PreparationStatus,
    RetrievalStatus,
)
from app.reviews.retrieval import ClauseRetriever
from app.reviews.service import ClauseRetrievalService


def clause() -> ContractClause:
    return ContractClause(
        clause_id="clause-0001",
        order=0,
        title="ค่าล่วงเวลา",
        text="ลูกจ้างทำงานล่วงเวลา",
        block_indices=[0],
        section_path=["ค่าล่วงเวลา"],
        boundary_reason=BoundaryReason.NUMBERED_CLAUSE,
    )


class ClauseRetrieverTests(unittest.IsolatedAsyncioTestCase):
    async def test_deduplicates_sources_and_preserves_query_provenance(self) -> None:
        def retrieve(**kwargs):
            score = 0.9 if "query one" in kwargs["query"] else 0.8
            return [
                {
                    "score": score,
                    "text": "กฎหมายเรื่องการทำงานล่วงเวลา",
                    "metadata": {
                        "law_name": "พระราชบัญญัติคุ้มครองแรงงาน พ.ศ. 2541",
                        "section": "มาตรา 24",
                        "source_file": "chap2.jsonl",
                    },
                }
            ]

        target = clause()
        queries = [
            ClauseQuery(
                query_id="clause-0001-q01",
                clause_id=target.clause_id,
                text="query one",
                block_indices=[0],
            ),
            ClauseQuery(
                query_id="clause-0001-q02",
                clause_id=target.clause_id,
                text="query two",
                block_indices=[0],
            ),
        ]

        results = await ClauseRetriever(retrieve).retrieve([target], queries)

        self.assertEqual(results[0].status, RetrievalStatus.READY)
        self.assertEqual(len(results[0].sources), 1)
        source = results[0].sources[0]
        self.assertTrue(source.source_id.startswith("law:"))
        self.assertEqual(source.best_score, 0.9)
        self.assertEqual(
            source.matched_query_ids,
            ["clause-0001-q01", "clause-0001-q02"],
        )
        self.assertEqual(len(source.matches), 2)

    async def test_marks_invalid_metadata_as_no_results(self) -> None:
        def retrieve(**kwargs):
            return [{"score": 0.5, "text": "ข้อความ", "metadata": {}}]

        target = clause()
        query = ClauseQuery(
            query_id="clause-0001-q01",
            clause_id=target.clause_id,
            text="query",
            block_indices=[0],
        )

        result = (await ClauseRetriever(retrieve).retrieve([target], [query]))[0]

        self.assertEqual(result.status, RetrievalStatus.NO_RESULTS)
        self.assertEqual(result.query_results[0].invalid_result_count, 1)
        self.assertEqual(result.sources, [])

    async def test_isolates_query_failures_and_returns_partial_clause(self) -> None:
        def retrieve(**kwargs):
            if kwargs["query"] == "fails":
                raise RuntimeError("private upstream detail")
            return [
                {
                    "score": 0.7,
                    "text": "ข้อความกฎหมาย",
                    "metadata": {"law_name": "กฎหมาย", "section": "มาตรา 1"},
                }
            ]

        target = clause()
        queries = [
            ClauseQuery(
                query_id="clause-0001-q01",
                clause_id=target.clause_id,
                text="works",
                block_indices=[0],
            ),
            ClauseQuery(
                query_id="clause-0001-q02",
                clause_id=target.clause_id,
                text="fails",
                block_indices=[0],
            ),
        ]

        result = (await ClauseRetriever(retrieve).retrieve([target], queries))[0]

        self.assertEqual(result.status, RetrievalStatus.PARTIAL)
        self.assertEqual(result.query_results[1].error_code, "RuntimeError")
        self.assertNotIn(
            "private upstream detail",
            result.model_dump_json(),
        )


class ClauseRetrievalServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepares_every_clause_with_existing_retriever_contract(self) -> None:
        calls: list[dict] = []

        def retrieve(**kwargs):
            calls.append(kwargs)
            return [
                {
                    "score": 0.75,
                    "text": "ข้อความกฎหมาย",
                    "metadata": {
                        "law_name": "พระราชบัญญัติคุ้มครองแรงงาน พ.ศ. 2541",
                        "section": "มาตรา 1",
                    },
                }
            ]

        document = ParsedDocument(
            document_id="doc-1",
            filename="contract.docx",
            blocks=[
                DocumentBlock(index=0, type="paragraph", text="ข้อ 1 ค่าจ้าง"),
                DocumentBlock(index=1, type="paragraph", text="ข้อ 2 เวลาทำงาน"),
            ],
        )

        result = await ClauseRetrievalService(retrieve).prepare(document, top_k=3)

        self.assertEqual(result.status, PreparationStatus.READY)
        self.assertEqual(result.summary.clause_count, 2)
        self.assertEqual(result.summary.query_count, 2)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call["top_k"] == 3 for call in calls))
        self.assertTrue(all(call["law_name"] is None for call in calls))

    async def test_does_not_call_retrieval_for_empty_document(self) -> None:
        calls = 0

        def retrieve(**kwargs):
            nonlocal calls
            calls += 1
            return []

        document = ParsedDocument(
            document_id="doc-empty",
            filename="empty.docx",
            blocks=[],
        )

        with self.assertRaises(ValueError):
            await ClauseRetrievalService(retrieve).prepare(document)
        self.assertEqual(calls, 0)


if __name__ == "__main__":
    unittest.main()


import unittest

from app.documents.models import DocumentBlock
from app.reviews.models import BoundaryReason, ContractClause
from app.reviews.queries import ClauseQueryBuilder
from app.reviews.segmentation import (
    ClauseSegmentationError,
    segment_contract,
    validate_clause_coverage,
)


def block(
    index: int,
    text: str,
    *,
    block_type: str = "paragraph",
    style: str | None = None,
) -> DocumentBlock:
    return DocumentBlock(index=index, type=block_type, text=text, style=style)


class ClauseSegmentationTests(unittest.TestCase):
    def test_builds_hierarchy_and_preserves_every_block_once(self) -> None:
        blocks = [
            block(0, "เงื่อนไขการจ้าง", block_type="heading", style="Heading 1"),
            block(1, "ข้อ 1 ค่าจ้าง"),
            block(2, "นายจ้างจ่ายค่าจ้างเดือนละ 20,000 บาท"),
            block(3, "1.1 ค่าล่วงเวลา"),
            block(4, "ชั่วโมง | อัตรา", block_type="table"),
        ]

        clauses = segment_contract(blocks)

        self.assertEqual(len(clauses), 2)
        self.assertEqual(clauses[0].block_indices, [0, 1, 2])
        self.assertEqual(clauses[0].number, "1")
        self.assertEqual(
            clauses[0].boundary_reason,
            BoundaryReason.HEADING_AND_NUMBERED,
        )
        self.assertEqual(clauses[1].block_indices, [3, 4])
        self.assertEqual(clauses[1].number, "1.1")
        self.assertEqual(clauses[1].parent_clause_id, clauses[0].clause_id)
        self.assertEqual(
            [index for clause in clauses for index in clause.block_indices],
            [0, 1, 2, 3, 4],
        )

    def test_normalizes_thai_clause_numbers(self) -> None:
        clauses = segment_contract(
            [
                block(0, "ข้อ ๑ การจ้างงาน"),
                block(1, "๑.๑ ระยะทดลองงาน"),
            ]
        )

        self.assertEqual([clause.number for clause in clauses], ["1", "1.1"])
        self.assertEqual(clauses[1].parent_clause_id, clauses[0].clause_id)

    def test_accepts_bare_number_only_for_structural_heading(self) -> None:
        clauses = segment_contract(
            [
                block(
                    0,
                    "1 ตำแหน่งและการเริ่มงาน",
                    block_type="heading",
                    style="Heading 1",
                ),
                block(1, "รายละเอียดตำแหน่งงาน"),
                block(2, "2 มกราคมไม่ใช่หัวข้อสัญญา"),
            ]
        )

        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0].number, "1")
        self.assertEqual(clauses[0].block_indices, [0, 1, 2])

    def test_keeps_preamble_before_first_numbered_clause(self) -> None:
        clauses = segment_contract(
            [
                block(0, "สัญญาฉบับนี้ทำขึ้นระหว่างนายจ้างและลูกจ้าง"),
                block(1, "ข้อ 1 ตำแหน่งงาน"),
            ]
        )

        self.assertEqual(len(clauses), 2)
        self.assertEqual(clauses[0].block_indices, [0])
        self.assertEqual(
            clauses[0].boundary_reason,
            BoundaryReason.DOCUMENT_START,
        )
        self.assertEqual(clauses[1].block_indices, [1])

    def test_uses_each_block_as_fallback_when_document_has_no_structure(self) -> None:
        clauses = segment_contract(
            [
                block(0, "นายจ้างตกลงจ้างลูกจ้าง"),
                block(1, "ลูกจ้างได้รับค่าจ้างทุกสิ้นเดือน"),
            ]
        )

        self.assertEqual(len(clauses), 2)
        self.assertEqual(
            [clause.boundary_reason for clause in clauses],
            [BoundaryReason.UNSTRUCTURED_BLOCK] * 2,
        )
        self.assertEqual(
            [clause.block_indices for clause in clauses],
            [[0], [1]],
        )

    def test_rejects_empty_documents(self) -> None:
        with self.assertRaises(ClauseSegmentationError):
            segment_contract([])

    def test_coverage_validator_rejects_duplicate_and_missing_blocks(self) -> None:
        blocks = [block(0, "หนึ่ง"), block(1, "สอง")]
        invalid_clause = ContractClause(
            clause_id="clause-0001",
            order=0,
            text="หนึ่ง",
            block_indices=[0, 0],
            boundary_reason=BoundaryReason.DOCUMENT_START,
        )

        with self.assertRaisesRegex(
            ClauseSegmentationError,
            r"missing=\[1\].*duplicates=\[0\]",
        ):
            validate_clause_coverage(blocks, [invalid_clause])


class ClauseQueryBuilderTests(unittest.TestCase):
    def test_builds_at_least_one_query_for_every_clause(self) -> None:
        blocks = [block(0, "ข้อ 1 ค่าจ้าง"), block(1, "จ่ายทุกสิ้นเดือน")]
        clause = segment_contract(blocks)[0]

        queries = ClauseQueryBuilder().build(clause, blocks)

        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0].clause_id, clause.clause_id)
        self.assertEqual(queries[0].block_indices, [0, 1])
        self.assertIn("ค่าจ้าง", queries[0].text)

    def test_splits_long_clauses_without_losing_block_provenance(self) -> None:
        blocks = [
            block(0, "ข้อ 1 " + "ก" * 400),
            block(1, "ข" * 400),
        ]
        clause = segment_contract(blocks)[0]

        queries = ClauseQueryBuilder(max_query_chars=256).build(clause, blocks)

        self.assertGreater(len(queries), 1)
        covered_indices = {
            index for query in queries for index in query.block_indices
        }
        self.assertEqual(covered_indices, {0, 1})
        self.assertEqual(
            [query.query_id for query in queries],
            [f"{clause.clause_id}-q{index:02d}" for index in range(1, len(queries) + 1)],
        )


if __name__ == "__main__":
    unittest.main()


import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.documents.models import ParsedDocument  # noqa: E402
from app.documents.parser import parse_docx  # noqa: E402
from app.reviews.models import PreparationStatus  # noqa: E402
from app.reviews.queries import ClauseQueryBuilder  # noqa: E402
from app.reviews.segmentation import segment_contract  # noqa: E402
from app.reviews.service import ClauseRetrievalService  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "tmp" / "clause-retrieval" / f"{timestamp}-{uuid4().hex[:8]}"


async def run(args: argparse.Namespace) -> int:
    input_path = args.input.resolve()
    output_dir = (args.output_dir or _default_output_dir()).resolve()
    if not input_path.is_file():
        print(f"Input DOCX does not exist: {input_path}", file=sys.stderr)
        return 1
    if input_path.suffix.lower() != ".docx":
        print(f"Input must be a .docx file: {input_path}", file=sys.stderr)
        return 1
    if output_dir.exists():
        print(f"Output directory already exists: {output_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True)
    document = ParsedDocument(
        document_id=str(uuid4()),
        filename=input_path.name,
        blocks=parse_docx(input_path.read_bytes()),
    )
    clauses = segment_contract(document.blocks)
    query_builder = ClauseQueryBuilder(max_query_chars=args.max_query_chars)
    queries = [
        query
        for clause in clauses
        for query in query_builder.build(clause, document.blocks)
    ]

    _write_json(output_dir / "document.json", document.model_dump(mode="json"))
    _write_json(
        output_dir / "clauses.json",
        [clause.model_dump(mode="json") for clause in clauses],
    )
    _write_json(
        output_dir / "queries.json",
        [query.model_dump(mode="json") for query in queries],
    )

    service = ClauseRetrievalService(
        max_concurrency=args.max_concurrency,
        max_query_chars=args.max_query_chars,
    )
    result = await service.prepare(document, top_k=args.top_k)
    _write_json(output_dir / "result.json", result.model_dump(mode="json"))
    _write_json(
        output_dir / "run.json",
        {
            "status": result.status.value,
            "input": str(input_path),
            "output_dir": str(output_dir),
            "summary": result.summary.model_dump(mode="json"),
        },
    )

    print(json.dumps(result.summary.model_dump(mode="json"), ensure_ascii=False, indent=2))
    print(f"Artifacts: {output_dir}")
    if result.status == PreparationStatus.READY:
        return 0
    if result.status == PreparationStatus.PARTIAL:
        return 2
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic clause segmentation and legal retrieval for a DOCX."
    )
    parser.add_argument("input", type=Path, help="Path to the input DOCX file")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-query-chars", type=int, default=4000)
    parser.add_argument("--max-concurrency", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))


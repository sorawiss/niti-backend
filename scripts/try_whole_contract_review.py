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
from app.llm.factory import create_llm_gateway  # noqa: E402
from app.reviews.review_models import ReviewRunStatus  # noqa: E402
from app.reviews.service import ClauseRetrievalService  # noqa: E402
from app.reviews.whole_contract_workflow import (  # noqa: E402
    WholeContractReviewWorkflow,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "tmp" / "whole-contract-review" / (
        f"{timestamp}-{uuid4().hex[:8]}"
    )


async def run(args: argparse.Namespace) -> int:
    input_path = args.input.resolve()
    output_dir = (args.output_dir or _default_output_dir()).resolve()
    if not input_path.is_file() or input_path.suffix.lower() != ".docx":
        print(f"Input must be an existing .docx file: {input_path}", file=sys.stderr)
        return 1
    if output_dir.exists():
        print(f"Output directory already exists: {output_dir}", file=sys.stderr)
        return 1

    document = ParsedDocument(
        document_id=str(uuid4()),
        filename=input_path.name,
        blocks=parse_docx(input_path.read_bytes()),
    )
    gateway = create_llm_gateway()
    output_dir.mkdir(parents=True)
    try:
        preparation = await ClauseRetrievalService(
            max_concurrency=args.retrieval_concurrency,
            max_query_chars=args.max_query_chars,
        ).prepare(document, top_k=args.top_k)
        review = await WholeContractReviewWorkflow(
            gateway,
            max_output_tokens=args.max_output_tokens,
        ).review(preparation, debug=True)
    finally:
        await gateway.close()

    _write_json(output_dir / "document.json", document.model_dump(mode="json"))
    _write_json(
        output_dir / "retrieval.json",
        preparation.model_dump(mode="json"),
    )
    _write_json(
        output_dir / "review.json",
        review.model_dump(mode="json", exclude={"llm_debug", "workflow_debug"}),
    )
    _write_json(
        output_dir / "llm-response.json",
        review.llm_debug.model_dump(mode="json"),
    )
    _write_json(
        output_dir / "workflow-debug.json",
        review.model_dump(mode="json")["workflow_debug"],
    )
    print(f"Workflow trace: {output_dir / 'workflow-debug.json'}")
    print(json.dumps(review.summary.model_dump(mode="json"), ensure_ascii=False, indent=2))
    print(f"Complete LLM response: {output_dir / 'llm-response.json'}")
    if review.status == ReviewRunStatus.READY:
        return 0
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan legal evidence searches and review a whole DOCX contract."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-query-chars", type=int, default=4000)
    parser.add_argument("--retrieval-concurrency", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
